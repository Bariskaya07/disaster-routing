# Demo Veri Paketi

Bu klasör jüri önünde tek tuşla açılacak hazır demo sahnesi içindir.

Beklenen dosyalar:

- `pre_disaster.png`
- `post_disaster.png`
- `metadata.json`

`metadata.json` örnek şeması:

```json
{
  "scene_id": "ankara-demo-01",
  "bbox": [32.8000, 39.8500, 32.8100, 39.8600],
  "start": [39.8510, 32.8010],
  "goal": [39.8590, 32.8090]
}
```

Not:
- Demo butonu bu üç dosya aynı anda mevcutsa aktif şekilde çalışır.
- `data/demo_library/` klasörü aday sahnelerin kütüphanesidir; burası ise yalnızca aktif sahne içindir.
- xView2 `label.json` dosyaları uygulamanın beklediği `metadata.json` ile aynı değildir.
- Jüri sunumunda fumble yaşamamak için sunumdan önce seçtiğiniz tek sahneyi bu klasöre aktive edin.
