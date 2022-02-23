arch=('r50' 'mbv2' 'reg600' 'reg800')
for((i=0;i<4;i++))
do 
    cd ${arch[i]}
    tmux new -s ${arch[i]} -d ./run.sh
    cd ..
done
