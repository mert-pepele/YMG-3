import os

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg") 
import matplotlib.pyplot as plt  

CIKTI_KLASORU = "ciktilar"
os.makedirs(CIKTI_KLASORU, exist_ok=True)
rapor_satirlari = []


def yaz(*parcalar):
    """Hem ekrana hem de sonuç dosyasına yazar."""
    satir = " ".join(str(p) for p in parcalar)
    print(satir)
    rapor_satirlari.append(satir)


tf.keras.utils.set_random_seed(42)

conv = tf.keras.layers.Conv2D(
    filters=16,
    kernel_size=(3, 3),
    activation="relu",
    kernel_initializer="he_normal",
)

dummy_input = tf.random.normal((1, 128, 128, 1))
output = conv(dummy_input)
weights, biases = conv.get_weights()

yaz("=== Bölüm 3: Konvolüsyon ===")
yaz("Giriş boyutu:", dummy_input.shape)
yaz("Çıkış boyutu:", output.shape)
yaz("Filtre ağırlıklarının boyutu:", weights.shape)
yaz("Bias boyutu:", biases.shape)
yaz("İlk filtre:")
yaz(np.array2string(weights[:, :, 0, 0], precision=4))


giris_kenari, kernel_kenari, adim = 128, 3, 1
beklenen_kenar = (giris_kenari - kernel_kenari) // adim + 1
yaz("\nFormülle beklenen çıkış kenarı: "
    f"({giris_kenari} - {kernel_kenari}) / {adim} + 1 = {beklenen_kenar}")
yaz("TensorFlow çıkışıyla aynı mı?:",
    output.shape[1] == beklenen_kenar and output.shape[2] == beklenen_kenar)

agirlik_sayisi = weights.size
bias_sayisi = biases.size
yaz(f"Ağırlık sayısı: 3*3*1*16 = {agirlik_sayisi}")
yaz(f"Bias sayısı: {bias_sayisi}")
yaz(f"Toplam parametre: {agirlik_sayisi + bias_sayisi} "
    f"(Keras count_params: {conv.count_params()})")


satir, sutun, filtre_no = 10, 20, 5
giris_np = dummy_input.numpy()[0, :, :, 0]
pencere = giris_np[satir:satir + 3, sutun:sutun + 3]       
carpim_toplami = np.sum(pencere * weights[:, :, 0, filtre_no])
bias_eklenmis = carpim_toplami + biases[filtre_no]
elle_sonuc = max(0.0, bias_eklenmis)                        
tf_sonuc = output.numpy()[0, satir, sutun, filtre_no]

yaz(f"\nElle kontrol (konum=({satir},{sutun}), filtre={filtre_no}):")
yaz("  3x3 pencere:\n", np.array2string(pencere, precision=3))
yaz(f"  Çarpımların toplamı : {carpim_toplami:.6f}")
yaz(f"  + bias              : {bias_eklenmis:.6f}")
yaz(f"  ReLU sonrası (elle) : {elle_sonuc:.6f}")
yaz(f"  TensorFlow çıktısı  : {tf_sonuc:.6f}")
yaz("  Eşit mi? (1e-5 tolerans):", bool(np.isclose(elle_sonuc, tf_sonuc, atol=1e-5)))


fan_in = kernel_kenari * kernel_kenari * 1
teorik_std = np.sqrt(2.0 / fan_in)
yaz(f"\nhe_normal: fan_in = {fan_in}, teorik std = sqrt(2/{fan_in}) = {teorik_std:.4f}")
yaz(f"Üretilen 144 ağırlığın ortalaması = {weights.mean():.4f}, std = {weights.std():.4f}")
yaz(f"En küçük / en büyük ağırlık = {weights.min():.4f} / {weights.max():.4f}")
yaz("Not: 144 örnek az olduğu için std teorik değere ancak yaklaşık olarak eşittir.")
yaz(f"Bias başlangıç değerleri (hepsi sıfır olmalı): {np.unique(biases)}")


sinir = np.abs(weights).max()
fig, eksenler = plt.subplots(2, 8, figsize=(16, 4.6))
for i, eksen in enumerate(eksenler.flat):
    filtre = weights[:, :, 0, i]
    goruntu = eksen.imshow(filtre, cmap="coolwarm", vmin=-sinir, vmax=sinir)
    for (r, c), deger in np.ndenumerate(filtre):
        eksen.text(c, r, f"{deger:.2f}", ha="center", va="center", fontsize=7)
    eksen.set_title(f"Filtre {i}", fontsize=9)
    eksen.set_xticks([])
    eksen.set_yticks([])
fig.suptitle("Eğitim öncesi 16 filtrenin başlangıç ağırlıkları (he_normal, seed=42)")
fig.colorbar(goruntu, ax=eksenler.ravel().tolist(), shrink=0.8, label="Ağırlık")
sekil_yolu = os.path.join(CIKTI_KLASORU, "bolum3_filtreler.png")
fig.savefig(sekil_yolu, dpi=150, bbox_inches="tight")
plt.close(fig)
yaz(f"\nFiltre görseli kaydedildi: {sekil_yolu}")


metin_yolu = os.path.join(CIKTI_KLASORU, "bolum3_sonuclar.txt")
with open(metin_yolu, "w", encoding="utf-8") as f:
    f.write("\n".join(rapor_satirlari) + "\n")
print(f"Sonuçlar kaydedildi: {metin_yolu}")
