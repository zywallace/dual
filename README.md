# Team
Yu Zhao, Fan Yang, Zikun Chen
for l in en fr; do for f in data/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done

for l in en fr; do for f in data/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
