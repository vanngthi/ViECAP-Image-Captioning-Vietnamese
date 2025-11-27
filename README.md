# Vietnamese Image Captioning use Traditional Detection Model

### Download data
Dự án sử dụng 2 bộ dữ liệu:
1. Dataset UIT-ViIC:
- Chạy lệnh download annotations data: 
```bash
curl -L -o viecap.zip 'https://drive.google.com/uc?export=download&id=1YexKrE6o0UiJhFWpE8M5LKoe6-k3AiM4'
```
- Unzip file viecap.zip
- Chạy lệnh download image:
```bash
python dataset/download_dataset.py
```
2. Dataset Flickr Sport
- Sử dụng data được download từ: github Perceval-Wilhelm/Image-captioning-convert-to-Vietnamese
```bash
git clone --no-checkout https://github.com/Perceval-Wilhelm/Image-captioning-convert-to-Vietnamese.git
cd Image-captioning-convert-to-Vietnamese
git sparse-checkout init --cone
git sparse-checkout set Dataset/Flick_sportball
git checkout main
```
3. Đồng nhất định dạng dữ liệu (COCO format)
```bash
python dataset/flick2coco.py
```
4. Tổ chức thư mục chứa dữ liệu:
```text
dataset/
├── Flick_sportball/
│   ├── image/
│   ├── train.json
│   └── test.json
│
└── UIT-ViIC/
    ├── image/
    ├── uitviic_captions_train2017.json
    ├── uitviic_captions_val2017.json
    └── uitviic_captions_test2017.json
```

Thống kê dữ liệu:
<div align="center">

- **UIT-ViIC**

  | Dataset | Ảnh | Caption |
  |:--------:|:---:|:-------:|
  | Train    | 3619 | 18101 |
  | Test     | 231  | 1155  |

- **Flickr_sportballs**

  | Dataset | Ảnh | Caption |
  |:--------:|:---:|:-------:|
  | Train    | 100 | 500 |
  | Test     | 100 | 500 |

</div>

### Prepare
1. Trích xuất thực thể trong tập training, tạo file pickle lưu [[entities], caption]
2. Embedding caption, tạo file pickle lưu [[entities], caption, caption_embedding]

### Training
3. bash train.sh
### Eval
4. bash eval.sh

