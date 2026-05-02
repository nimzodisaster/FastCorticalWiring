python  run_pipeline.py     subjectlist.csv     --subjects-dir /mnt/gold/master_fs_bibsnet/SUBJECTS  \
--surf-type pial.cortexonly.qd.n60000    --jobs 20  --no-mask --scales 0.0005 0.0008, .001 .002 .005 .01 .015 --n-samples-between-scales 10 --boundary-cap-fraction 0.5 --output-label geodesics01 --force-logical --overwrite


python fastcw_derived_metrics_to_schaefer.py \
  --subjects-dir /mnt/gold/master_fs_bibsnet/SUBJECTS \
  --subjectlist /data/users/jlee38/homefolders/Documents/FastCorticalWiring/FastCorticalWiring/one_off_scripts/subjectlist.csv \
  --metric-stem pial.cortexonly.qd.n60000 \
  --output-label lausanne5 \
  --surf-name pial \
  --scale 0.005 0.01 .05 \
  --annot-stem fsaverage_atlas-Lausanne2018_scale-5_dseg.label \
  --output-types derived source trad \
  --output-dir ./fastcw_derived_lausanne5_csv \
  --overwrite-surfaces \
  --overwrite-csv
