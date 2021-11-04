import math
from typing import Tuple

import torch
from torch.quantization.observer import _ObserverBase

from mqbench.utils import sync_tensor, pot_quantization, is_symmetric_quant


class ObserverBase(_ObserverBase):
    '''
        Support per-tensor / per-channel.
        dtype: quant min/max can be infered using dtype, we actually do not need this.
        qscheme: quantization scheme
        reduce_range: special for fbgemm to avoid overflow
        quant_min: fix point value min
        quant_max: fix point value max
        ch_axis: per-channel axis or per-tensor(-1)
        above is similiar to torch observer.
        pot_scale: indecate wheather scale is power of two.
    '''

    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False):
        super(ObserverBase, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max)
        self.ch_axis = ch_axis
        self.pot_scale = pot_scale
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    @torch.jit.export
    def calculate_qparams(self) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Calculates the quantization parameters."""
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        if self.pot_scale:
            scale = pot_quantization(scale)
        scale.data = sync_tensor(scale).data
        zero_point.data = sync_tensor(zero_point).data
        return scale, zero_point

    @torch.jit.export
    def _calculate_qmin_qmax(self) -> Tuple[int, int]:
        r"""Calculates actual qmin and qmax based on the quantization range,
        observer datatype and if range is reduced.
        """
        if self.has_customized_qrange:
            # This initialization here is to be resolve TorchScript compilation issues and allow
            # using of refinement to decouple initial_qmin and initial_qmax from quantization range.
            # The actual values of initial_qmin and initial_qmax will be reset below.
            initial_quant_min, initial_quant_max = 0, 255
            # The following assignment of self.qmin and self.qmax to the local variables and the if check refine the
            # attribute from Optional valid integers for use, based on TorchScript's requirements.
            custom_quant_min, custom_quant_max = self.quant_min, self.quant_max
            if custom_quant_min is not None and custom_quant_max is not None:
                initial_quant_min, initial_quant_max = (
                    custom_quant_min,
                    custom_quant_max,
                )

            qrange_len = initial_quant_max - initial_quant_min + 1
            if is_symmetric_quant(self.qscheme):
                quant_min, quant_max = -qrange_len // 2, qrange_len // 2 - 1
            else:
                quant_min, quant_max = 0, qrange_len - 1
            if self.reduce_range:
                quant_min, quant_max = quant_min // 2, quant_max // 2
        else:
            # Fallback onto default 8-bit qmin and qmax calculation if dynamic range is not used.
            if self.dtype == torch.qint8:
                if self.reduce_range:
                    quant_min, quant_max = -64, 63
                else:
                    quant_min, quant_max = -128, 127
            elif self.dtype == torch.quint8:
                if self.reduce_range:
                    quant_min, quant_max = 0, 127
                else:
                    quant_min, quant_max = 0, 255
            else:
                quant_min, quant_max = 0, 15
        return quant_min, quant_max

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={} ch_axis={} pot={}".format(self.min_val if self.ch_axis == -1 else 'List', 
                                                                 self.max_val if self.ch_axis == -1 else 'List',
                                                                 self.ch_axis, self.pot_scale)


class MinMaxObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False):
        super(MinMaxObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max, ch_axis, pot_scale)

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        self.min_val = torch.min(self.min_val, min_val_cur)
        self.max_val = torch.max(self.max_val, max_val_cur)

        return x


class MinMaxFloorObserver(ObserverBase):
    '''
    Calculate minmax of whole calibration dataset.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine,
                 reduce_range=False, quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False):
        super(MinMaxFloorObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max, ch_axis, pot_scale)

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        self.min_val = min_val_cur
        self.max_val = max_val_cur
        self._x = x 
        return x

    def calculate_qparams(self):
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = scale.data * 0 + max(self.min_val / self.quant_min, self.max_val / self.quant_max) 
        if abs(scale) < 2 ** -12:
            final_scale = 0 * scale
        else:
            max_scale = 1 / scale 
            max_scale = torch.floor(max_scale.log2())
            min_loss = torch.tensor([float('inf')])
            final_scale = max_scale
            max_scale = int(max_scale)
            for s in range(max_scale - 1, max_scale + 5):
                _s = 1 / 2 ** s 
                new_x = _s * torch.clamp(torch.floor(self._x / _s), self.quant_min, self.quant_max)
                loss = ((new_x - self._x)**2).sum()
                min_loss = min_loss.to(loss.device)
                if loss < min_loss:
                    min_loss = loss
                    final_scale = s  
        scale = scale.data * 0 + 1 / (2 ** final_scale)
        zero_point = torch.zeros_like(zero_point)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point


class EMAMinMaxObserver(ObserverBase):
    """Moving average min/max among batches.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, ema_ratio=0.9):
        super(EMAMinMaxObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                ch_axis, pot_scale)
        self.ema_ratio = ema_ratio

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (1.0 - self.ema_ratio)
        return x

class EMAMinMaxFloorObserver(ObserverBase):
    """Moving average min/max among batches.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, ema_ratio=0.9):
        super(EMAMinMaxFloorObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                     ch_axis, pot_scale)
        self.ema_ratio = ema_ratio

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        self._x = x
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = min_val_cur
            self.max_val = max_val_cur
        else:
            self.min_val = self.min_val * self.ema_ratio + min_val_cur * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + max_val_cur * (1.0 - self.ema_ratio)
        return x

    def calculate_qparams(self):
        scale, zero_point = self._calculate_qparams(self.min_val, self.max_val)
        scale.data = scale.data * 0 + max(self.min_val / self.quant_min, self.max_val / self.quant_max) 
        if abs(scale) < 2 ** -12:
            final_scale = 0 * scale
        else:
            max_scale = 1 / scale 
            max_scale = torch.floor(max_scale.log2())
            min_loss = torch.tensor([float('inf')])
            final_scale = max_scale
            max_scale = int(max_scale)
            for s in range(max_scale - 1, max_scale + 5):
                _s = 1 / 2 ** s 
                new_x = _s * torch.clamp(torch.floor(self._x / _s), self.quant_min, self.quant_max)
                loss = ((new_x - self._x)**2).sum()
                min_loss = min_loss.to(loss.device)
                if loss < min_loss:
                    min_loss = loss
                    final_scale = s  
        scale = scale.data * 0 + 1 / (2 ** final_scale)
        zero_point = torch.zeros_like(zero_point)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point


class EMAQuantileObserver(ObserverBase):
    """Moving average quantile among batches.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, ema_ratio=0.9, threshold=0.99999, bins=2048):
        super(EMAQuantileObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max,
                                                  ch_axis, pot_scale)
        assert self.ch_axis == -1, "Quantile observer only support in per-tensor scheme."
        self.ema_ratio = ema_ratio
        self.threshold = threshold
        self.bins = bins

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch._aminmax(x)
        hist = torch.histc(torch.abs(x), bins=self.bins, min=0., max=torch.max(-min_val_cur, max_val_cur))
        cur_total = 0
        clip_value = torch.max(-min_val_cur, max_val_cur)
        for i, cnt in enumerate(hist):
            if cur_total + cnt >= self.threshold * x.numel():
                clip_value = (i + 0.5) * (max_val_cur / self.bins)
                break

        if self.max_val.numel() <= 1 and self.max_val.isinf():
            self.min_val = max(min_val_cur, -clip_value)
            self.max_val = min(max_val_cur, clip_value)
        else:
            self.min_val = self.min_val * self.ema_ratio + max(min_val_cur, -clip_value) * (1.0 - self.ema_ratio)
            self.max_val = self.max_val * self.ema_ratio + min(max_val_cur, clip_value) * (1.0 - self.ema_ratio)
        return x


class ClipStdObserver(ObserverBase):
    """Clip std.
    """

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False, 
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False, std_scale=2.6):
        super(ClipStdObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max, ch_axis, pot_scale)
        self.std_scale = std_scale

    def forward(self, x_orig):
        r"""Records the running minimum and maximum of ``x``."""
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            min_val_cur, max_val_cur = torch._aminmax(x)
            mean = x.mean()
            std = x.std()
        else:
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            min_val_cur, max_val_cur = torch._aminmax(y, 1)
            mean = y.mean(1)
            std = y.std(1)

        # using statistics to clip min and max
        min_val = torch.minimum(mean - self.std_scale * std, min_val_cur)
        max_val = torch.maximum(mean + self.std_scale * std, max_val_cur)

        self.min_val = min_val
        self.max_val = max_val

        return x


class LSQObserver(ObserverBase):
    '''
    LSQ observer.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False):
        super(LSQObserver, self).__init__(dtype, qscheme, reduce_range, quant_min, quant_max, ch_axis, pot_scale)
        self.tensor_norm = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.tensor_norm = x.abs().mean()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.tensor_norm = y.abs().mean(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self):
        scale = 2 * self.tensor_norm / math.sqrt(self.quant_max)
        if self.pot_scale:
            scale = pot_quantization(scale)
        zero_point = torch.zeros_like(self.tensor_norm)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point


class LSQPlusObserver(ObserverBase):
    '''
    LSQ+ observer.
    '''

    def __init__(self, dtype=torch.quint8, qscheme=torch.per_tensor_affine, reduce_range=False,
                 quant_min=None, quant_max=None, ch_axis=-1, pot_scale=False):

        super(LSQPlusObserver, self).__init__(dtype, qscheme, reduce_range, 
                                              quant_min, quant_max, ch_axis, pot_scale)
        self.mean = None
        self.std = None

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.to(self.min_val.dtype)
        if self.ch_axis == -1:
            self.mean = x.mean()
            self.std = x.std()
            self.min_val, self.max_val = torch._aminmax(x)
        else:
            # compute channel-wise mean
            x_dim = x.size()
            new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
            new_axis_list[self.ch_axis] = 0
            new_axis_list[0] = self.ch_axis
            y = x.permute(new_axis_list)
            y = torch.flatten(y, start_dim=1)
            self.mean = y.mean(1)
            self.std = y.std(1)
            self.min_val, self.max_val = torch._aminmax(y)

        return x

    def calculate_qparams(self):
        scale = torch.maximum((self.mean - 3 * self.std).abs(),
                              (self.mean + 3 * self.std).abs()) / (self.quant_max - self.quant_min + 1)
        if self.pot_scale:
            scale = pot_quantization(scale)
        zero_point = torch.zeros_like(self.mean)
        if not is_symmetric_quant(self.qscheme):
            if self.min_val >= 0.:
                zero_point = self.quant_min - torch.round(self.min_val / scale)
        sync_tensor(scale)
        sync_tensor(zero_point)
        return scale, zero_point
