#!/bin/bash

directory="../data/VCTK_wav"
dir_out="../data/VCTK_cat_train"

if [ ! -d $dir_out ]
then
    mkdir $dir_out
fi

for dir in `ls $directory`
do
    if [ -d $directory/$dir ]
    then
        if [ $dir != "p315" ]
        then
            echo $dir
            ls $directory/$dir | wc -l
            mkdir $dir_out/$dir
            for file in `ls $directory/$dir`
            do
                if [ -f $directory/$dir/$file ]
                then
                    if [ "${file##*.}"x = "wav"x ]
                    then
                        finfo=(${file//./ })
                        name=${finfo[0]}
                        info=(${name//_/ })
                        seq=`echo ${info[1]} | awk '{print int($0)}'`
                        s=$(( $seq % 3 ))
                        if [ ! $s -eq 0 ]
                        then
                            if [ ! -f $dir_out/$dir/${dir}.wav ]
                            then
                                cp $directory/$dir/$file $dir_out/$dir/${dir}.wav
                            else
                                sox $dir_out/$dir/${dir}.wav $directory/$dir/$file $dir_out/$dir/${dir}_tmp.wav
                                rm $dir_out/$dir/${dir}.wav
                                mv $dir_out/$dir/${dir}_tmp.wav $dir_out/$dir/${dir}.wav
                            fi
                        fi
                    fi
                fi
            done
            ls $dir_out/$dir | wc -l
        fi
    fi
done
