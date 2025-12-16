
python evaluate.py \
--edited_images_dir /apdcephfs_cq11/share_1603164/user/tianyuanqu/output/reason_gen/stage2_40step_qi_origin_global_sym_dec \
--output_json /apdcephfs_cq11/share_1603164/user/tianyuanqu/output/reason_gen/stage2_40step_qi_origin_global_sym_dec.json \
--limit 10
python evaluate.py \
--edited_images_dir /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/eval/reason_gen_coco/output/replan_flux_test \
--output_json /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/eval/reason_gen_coco/output/replan_flux_test.json \
--limit 10
python evaluate.py \
--edited_images_dir /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/eval/reason_gen_coco/output/kontext \
--output_json /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/eval/reason_gen_coco/output/kontext.json \
--limit 10
python evaluate.py \
--edited_images_dir /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/eval/reason_gen_coco/output/replan_qwen_image \
--output_json /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/eval/reason_gen_coco/output/replan_qwen_image.json \
--limit 10
python -m replan.eval.evaluate \
--edited_images_dir /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/output/replan_qwen_image_5 \
--output_json /apdcephfs_cq11/share_1603164/user/tianyuanqu/research/RePlan/output/replan_qwen_image_5.json \
--limit 10
