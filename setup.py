import setuptools
from mqbench import __version__


def read_requirements():
    reqs = []
    with open('requirements.txt', 'r') as fin:
        for line in fin.readlines():
            reqs.append(line.strip())
    return reqs


setuptools.setup(
    name="MQBench",
    version=__version__,
    author="The Great Cold",
    author_email="",
    description=("Quantization aware training."),
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    classifiers=(
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux"),
    install_requires=read_requirements()
)
