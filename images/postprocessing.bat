## run under images

## add prediction masks with tp yellow
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD3_test_img_cp_masks.png --saveas  cyto_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD4_test_img_cp_masks.png --saveas  cyto_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD8_test_img_cp_masks.png --saveas  cyto_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position9_CD3_test_img_cp_masks.png --saveas  cyto_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position10_CD3_test_img_cp_masks.png --saveas  cyto_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 test_cytomasks/M926910_CFL_Position7_CD3_test_img_cp_masks.png --saveas  cyto_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 test_cytomasks/M926910_CFL_Position13_CD3_test_img_cp_masks.png --saveas  cyto_tp

python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 test_cytotrain7masks/M872956_JML_Position8_CD3_test_img_cp_masks.png --saveas  cytotrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 test_cytotrain7masks/M872956_JML_Position8_CD4_test_img_cp_masks.png --saveas  cytotrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 test_cytotrain7masks/M872956_JML_Position8_CD8_test_img_cp_masks.png --saveas  cytotrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 test_cytotrain7masks/M872956_JML_Position9_CD3_test_img_cp_masks.png --saveas  cytotrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 test_cytotrain7masks/M872956_JML_Position10_CD3_test_img_cp_masks.png --saveas  cytotrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 test_cytotrain7masks/M926910_CFL_Position7_CD3_test_img_cp_masks.png --saveas  cytotrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 test_cytotrain7masks/M926910_CFL_Position13_CD3_test_img_cp_masks.png --saveas  cytotrain7_tp

python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 test_tnnucleartrain7masks/M872956_JML_Position8_CD3_test_dc_masks.png --saveas  tnnucleartrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 test_tnnucleartrain7masks/M872956_JML_Position8_CD4_test_dc_masks.png --saveas  tnnucleartrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 test_tnnucleartrain7masks/M872956_JML_Position8_CD8_test_dc_masks.png --saveas  tnnucleartrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 test_tnnucleartrain7masks/M872956_JML_Position9_CD3_test_dc_masks.png --saveas  tnnucleartrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 test_tnnucleartrain7masks/M872956_JML_Position10_CD3_test_dc_masks.png --saveas  tnnucleartrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 test_tnnucleartrain7masks/M926910_CFL_Position7_CD3_test_dc_masks.png --saveas  tnnucleartrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 test_tnnucleartrain7masks/M926910_CFL_Position13_CD3_test_dc_masks.png --saveas  tnnucleartrain7_tp

python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 test_jacstrain7masks/M872956_JML_Position8_CD3_test_mrmasks.png --saveas  jacstrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 test_jacstrain7masks/M872956_JML_Position8_CD4_test_mrmasks.png --saveas  jacstrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 test_jacstrain7masks/M872956_JML_Position8_CD8_test_mrmasks.png --saveas  jacstrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 test_jacstrain7masks/M872956_JML_Position9_CD3_test_mrmasks.png --saveas  jacstrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 test_jacstrain7masks/M872956_JML_Position10_CD3_test_mrmasks.png --saveas  jacstrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 test_jacstrain7masks/M926910_CFL_Position7_CD3_test_mrmasks.png --saveas  jacstrain7_tp
python -m syotil colortp --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 test_jacstrain7masks/M926910_CFL_Position13_CD3_test_mrmasks.png --saveas  jacstrain7_tp





## add gt and cyto masks
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD3_test_img_cp_masks.png --saveas gt-cyto
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD4_test_img_cp_masks.png --saveas gt-cyto
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD8_test_img_cp_masks.png --saveas gt-cyto
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position9_CD3_test_img_cp_masks.png --saveas gt-cyto
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position10_CD3_test_img_cp_masks.png --saveas gt-cyto
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 test_cytomasks/M926910_CFL_Position7_CD3_test_img_cp_masks.png --saveas gt-cyto
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 test_cytomasks/M926910_CFL_Position13_CD3_test_img_cp_masks.png --saveas gt-cyto


## add gt and train7 masks
python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 test_cytotrain7masks/M872956_JML_Position8_CD3_test_img_cp_masks.png --saveas gt-train7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 predictions-train7/M872956_JML_Position8_CD4_test_img_cp_masks.png --saveas gt-train7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 predictions-train7/M872956_JML_Position8_CD8_test_img_cp_masks.png --saveas gt-train7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 predictions-train7/M872956_JML_Position9_CD3_test_img_cp_masks.png --saveas gt-train7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 predictions-train7/M872956_JML_Position10_CD3_test_img_cp_masks.png --saveas gt-train7
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 predictions-train7/M926910_CFL_Position7_CD3_test_img_cp_masks.png --saveas gt-train7
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 predictions-train7/M926910_CFL_Position13_CD3_test_img_cp_masks.png --saveas gt-train7


## add gt masks
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --saveas gt
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --saveas gt
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --saveas gt
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --saveas gt
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --saveas gt
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --saveas gt
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --saveas gt


## add cyto masks
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD3_test_img_cp_masks.png --saveas cyto
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD4_test_img_cp_masks.png --saveas cyto
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 test_cytomasks/M872956_JML_Position8_CD8_test_img_cp_masks.png --saveas cyto
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position9_CD3_test_img_cp_masks.png --saveas cyto
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 test_cytomasks/M872956_JML_Position10_CD3_test_img_cp_masks.png --saveas cyto
#python -m syotil overlaymasks --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 test_cytomasks/M926910_CFL_Position7_CD3_test_img_cp_masks.png --saveas cyto
#python -m syotil overlaymasks --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 test_cytomasks/M926910_CFL_Position13_CD3_test_img_cp_masks.png --saveas cyto

## add train7 masks
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position8_CD3_test_img.png --mask2 predictions-train7/M872956_JML_Position8_CD3_test_img_cp_masks.png --saveas train7
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position8_CD4_test_img.png --mask2 predictions-train7/M872956_JML_Position8_CD4_test_img_cp_masks.png --saveas train7
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position8_CD8_test_img.png --mask2 predictions-train7/M872956_JML_Position8_CD8_test_img_cp_masks.png --saveas train7
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position9_CD3_test_img.png --mask2 predictions-train7/M872956_JML_Position9_CD3_test_img_cp_masks.png --saveas train7
#python -m syotil overlaymasks --imagefile test_images/M872956_JML_Position10_CD3_test_img.png --mask2 predictions-train7/M872956_JML_Position10_CD3_test_img_cp_masks.png --saveas train7
#python -m syotil overlaymasks --imagefile test_images/M926910_CFL_Position7_CD3_test_img.png --mask2 predictions-train7/M926910_CFL_Position7_CD3_test_img_cp_masks.png --saveas train7
#python -m syotil overlaymasks --imagefile test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 predictions-train7/M926910_CFL_Position13_CD3_test_img_cp_masks.png --saveas train7

## add gt and train7 masks
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD3_test_masks.png  --imagefile test_images/test_images/M872956_JML_Position8_CD3_test_img.png  --mask2 test_tnnucleartrain7masks/M872956_JML_Position8_CD3_test_dc_masks.png  --saveas gt-tnnucleartrain7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD4_test_masks.png  --imagefile test_images/test_images/M872956_JML_Position8_CD4_test_img.png  --mask2 test_tnnucleartrain7masks/M872956_JML_Position8_CD4_test_dc_masks.png  --saveas gt-tnnucleartrain7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position8_CD8_test_masks.png  --imagefile test_images/test_images/M872956_JML_Position8_CD8_test_img.png  --mask2 test_tnnucleartrain7masks/M872956_JML_Position8_CD8_test_dc_masks.png  --saveas gt-tnnucleartrain7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position9_CD3_test_masks.png  --imagefile test_images/test_images/M872956_JML_Position9_CD3_test_img.png  --mask2 test_tnnucleartrain7masks/M872956_JML_Position9_CD3_test_dc_masks.png  --saveas gt-tnnucleartrain7
#python -m syotil overlaymasks --mask1 test_gtmasks/M872956_JML_Position10_CD3_test_masks.png --imagefile test_images/test_images/M872956_JML_Position10_CD3_test_img.png --mask2 test_tnnucleartrain7masks/M872956_JML_Position10_CD3_test_dc_masks.png --saveas gt-tnnucleartrain7
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position7_CD3_test_masks.png  --imagefile test_images/test_images/M926910_CFL_Position7_CD3_test_img.png  --mask2 test_tnnucleartrain7masks/M926910_CFL_Position7_CD3_test_dc_masks.png  --saveas gt-tnnucleartrain7
#python -m syotil overlaymasks --mask1 test_gtmasks/M926910_CFL_Position13_CD3_test_masks.png --imagefile test_images/test_images/M926910_CFL_Position13_CD3_test_img.png --mask2 test_tnnucleartrain7masks/M926910_CFL_Position13_CD3_test_dc_masks.png --saveas gt-tnnucleartrain7

# get AP
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_060 > APresults/csi_tfmrcnn1_060.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_080 > APresults/csi_tfmrcnn1_080.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_100 > APresults/csi_tfmrcnn1_100.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_120 > APresults/csi_tfmrcnn1_120.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_140 > APresults/csi_tfmrcnn1_140.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_160 > APresults/csi_tfmrcnn1_160.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_180 > APresults/csi_tfmrcnn1_180.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn1_200 > APresults/csi_tfmrcnn1_200.txt

#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn_cellseg > APresults/csi_tfmrcnn_cellseg.txt
#python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_tfmrcnn_cellseg2 > APresults/csi_tfmrcnn_cellseg2.txt


python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --min_size 1
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --min_size 50
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --min_size 100 #331 cnt, mAP .76
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --min_size 150

python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytomasks --min_size 0
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytomasks --min_size 50
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytomasks --min_size 100
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytomasks --min_size 150

python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 1
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 2500
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 3500
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 4000 #333 cnt, mAP .77
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 5000
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 6000
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 10000
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_totalintensity 15000

python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_avgintensity 1
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_avgintensity 30
python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_avgintensity 35 # 336 cnt, mAP .73


python -m syotil checkprediction --gtfolder test_gtmasks --predfolder test_cytotrain7masks --imgfolder test_images --min_size 100 --min_totalintensity 4000 #331, mAP .78
