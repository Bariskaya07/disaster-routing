# Demo Kütüphanesi

Bu klasör, jüri sunumu için aday xView2 sahnelerini toplu halde saklamak içindir.

Önemli ayrım:

- `pre_label.json` ve `post_label.json` dosyaları xView2/xBD'nin özgün anotasyon dosyalarıdır.
- Bu dosyalar, uygulamanın beklediği sade `metadata.json` ile aynı şey değildir.
- Her sahne klasöründeki `metadata.json`, uygulama için ayrıca üretilmiş sade şemadır:
  - `bbox`
  - `start`
  - `goal`
  - `crs`

Önerilen kullanım:

1. Adayları `data/demo_library/` altında inceleyin.
2. Sunumda kullanmak istediğiniz tek sahneyi seçin.
3. O sahneyi `data/demo/` içine aktive edin.

Hazırlama komutu:

```bash
/home/bariskaya/Projelerim/UAV/venv/bin/python scripts/prepare_demo_library.py --activate-scene santa-rosa-wildfire_00000157
```

Aktivasyon komutu:

```bash
/home/bariskaya/Projelerim/UAV/venv/bin/python scripts/activate_demo_scene.py data/demo_library/santa-rosa-wildfire/santa-rosa-wildfire_00000157
```
