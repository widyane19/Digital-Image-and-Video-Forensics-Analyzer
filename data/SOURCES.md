# Sources of the public media

Written by `python -m src.external_datasets`. Our own photos and videos are not listed.

<!-- vision -->
## VISION

- Source: <https://lesc.dinfo.unifi.it/en/datasets>
- Paper: D. Shullani, M. Fontani, M. Iuliani, O. Al Shaya and A. Piva, "VISION: a video and image dataset
  for source identification", *EURASIP Journal on Information Security*, 2017.
  doi:10.1186/s13635-017-0067-2
- Licence: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- Used as extra unedited originals (in `originals/`, names starting `vision_`); their manipulated copies
  are made by `build_dataset.py`.
- Files taken:
  - `vision_D31_I_nat_0095.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D31_Samsung_GalaxyS4Mini/images/nat/D31_I_nat_0095.jpg>
  - `vision_D23_I_nat_0024.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D23_Asus_Zenfone2Laser/images/nat/D23_I_nat_0024.jpg>
  - `vision_D04_I_nat_0140.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D04_LG_D290/images/nat/D04_I_nat_0140.jpg>
  - `vision_D17_I_nat_0133.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D17_Microsoft_Lumia640LTE/images/nat/D17_I_nat_0133.jpg>
  - `vision_D21_I_nat_0170.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D21_Wiko_Ridge4G/images/nat/D21_I_nat_0170.jpg>
  - `vision_D07_I_nat_0039.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D07_Lenovo_P70A/images/nat/D07_I_nat_0039.jpg>
  - `vision_D20_I_nat_0100.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D20_Apple_iPadMini/images/nat/D20_I_nat_0100.jpg>
  - `vision_D32_I_nat_0150.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D32_OnePlus_A3003/images/nat/D32_I_nat_0150.jpg>
  - `vision_D16_I_nat_0007.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D16_Huawei_P9Lite/images/nat/D16_I_nat_0007.jpg>
  - `vision_D24_I_nat_0063.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D24_Xiaomi_RedmiNote3/images/nat/D24_I_nat_0063.jpg>
  - `vision_D12_I_nat_0101.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D12_Sony_XperiaZ1Compact/images/nat/D12_I_nat_0101.jpg>
  - `vision_D27_I_nat_0157.jpg` from <https://lesc.dinfo.unifi.it/VISION/dataset/D27_Samsung_GalaxyS5/images/nat/D27_I_nat_0157.jpg>
  - `vision_D27_V_flat_panrot_0002.mp4` from <https://lesc.dinfo.unifi.it/VISION/dataset/D27_Samsung_GalaxyS5/videos/flat/D27_V_flat_panrot_0002.mp4>
  - `vision_D17_V_outdoor_still_0001.mp4` from <https://lesc.dinfo.unifi.it/VISION/dataset/D17_Microsoft_Lumia640LTE/videos/outdoor/D17_V_outdoor_still_0001.mp4>
<!-- /vision -->

<!-- csvted -->
## CSVTED

- Source: <https://huggingface.co/datasets/CardiffVisualComputing/CSVTED>
- Paper: COMSATS Structured Video Tampering Evaluation Dataset (CSVTED), *Machine Vision and Applications*
  (post-print: <https://rosinpl.github.io/resources/papers/CSVTED-video-MVA-postprint.pdf>)
- Licence: not stated on the dataset page; check before publishing results.
- Used as a separate labelled set (`external/csvted/labels.csv`); the edits were made by the
  dataset's authors, not by us.
- Files taken:
  - `originals/Standing Person.AVI`
  - `frame_duplication/015_Dup_171_185_Standing Person.avi`
  - `frame_insertion/015_Ins_081_095_Standing Person.avi`
  - `frame_removal/049_Del_121_169_Standing Person.avi`
  - `originals/Customer Dealing(E).mp4`
  - `frame_duplication/010_Dup_179_188_Customer Dealing(E).mp4`
  - `frame_insertion/035_Ins_144_178_Customer Dealing(E).mp4`
  - `frame_removal/044_Del_041_084_Customer Dealing(E).mp4`
  - `originals/CCTV Customer Footage(E).mp4`
  - `frame_duplication/010_Dup_061_070_CCTV Customer Footage(E).mp4`
  - `frame_insertion/015_Ins_025_039_CCTV Customer Footage(E).mp4`
  - `frame_removal/043_Del_006_048_CCTV Customer Footage(E).mp4`
  - `originals/Children.mp4`
  - `frame_duplication/010_Dup_076_085_Children.mp4`
  - `frame_insertion/010_Ins_037_045_Children.mp4`
  - `frame_removal/053_Del_057_109_Children.mp4`
  - `originals/Black Car.AVI`
  - `frame_duplication/010_Dup_151_160_Black Car.avi`
  - `frame_insertion/010_Ins_033_042_Black Car.avi`
  - `frame_removal/034_Del_053_086_Black Car.avi`
<!-- /csvted -->
