import os

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle  

CIKTI_KLASORU = "ciktilar"
os.makedirs(CIKTI_KLASORU, exist_ok=True)
rapor_satirlari = []


def yaz(*parcalar):
    satir = " ".join(str(p) for p in parcalar)
    print(satir)
    rapor_satirlari.append(satir)



input_data = tf.constant([
    [[1., 3., 2., 4.],
     [5., 6., 1., 2.],
     [7., 2., 8., 3.],
     [4., 1., 5., 9.]]
])
input_data = tf.reshape(input_data, (1, 4, 4, 1))  

pool = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
output = pool(input_data)

giris = input_data.numpy()[0, :, :, 0]
tf_cikis = output.numpy()[0, :, :, 0]

yaz("=== Bölüm 4: Max Pooling ===")
yaz("Giriş:")
yaz(giris)
yaz("Pooling sonucu (TensorFlow):")
yaz(tf_cikis)


pencere, adim = 2, 2
cikis_kenari = (giris.shape[0] - pencere) // adim + 1
elle_cikis = np.zeros((cikis_kenari, cikis_kenari), dtype=giris.dtype)
yaz("\nElle hesaplama:")
for i in range(cikis_kenari):
    for j in range(cikis_kenari):
        bolge = giris[i * adim:i * adim + pencere, j * adim:j * adim + pencere]
        elle_cikis[i, j] = bolge.max()
        yaz(f"  Bölge ({i},{j}) = {bolge.flatten().tolist()} -> en büyük: {bolge.max():g}")
yaz("Elle bulunan sonuç:")
yaz(elle_cikis)
yaz("TensorFlow ile aynı mı?:", bool(np.array_equal(elle_cikis, tf_cikis)))
yaz(f"Giriş tensör şekli: {tuple(input_data.shape)}, çıkış tensör şekli: {tuple(output.shape)}")
yaz(f"Örnek başına değer sayısı: {giris.size} -> {tf_cikis.size}")
yaz(f"Pooling katmanının öğrenilebilir parametre sayısı: {pool.count_params()}")


renkler = ["#cfe2f3", "#d9ead3", "#fff2cc", "#f4cccc"]
fig, (sol, sag) = plt.subplots(1, 2, figsize=(9, 4.2),
                               gridspec_kw={"width_ratios": [4, 2.6]})
for i in range(4):
    for j in range(4):
        bolge_no = (i // 2) * 2 + (j // 2)
        en_buyuk_mu = giris[i, j] == elle_cikis[i // 2, j // 2]
        sol.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor=renkler[bolge_no],
                                edgecolor="gray"))
        sol.text(j + 0.5, 3 - i + 0.5, f"{giris[i, j]:g}", ha="center", va="center",
                 fontsize=16, fontweight="bold" if en_buyuk_mu else "normal",
                 color="black" if en_buyuk_mu else "#666666")

sol.plot([2, 2], [0, 4], color="black", lw=2.5)
sol.plot([0, 4], [2, 2], color="black", lw=2.5)
sol.set_xlim(0, 4); sol.set_ylim(0, 4); sol.set_aspect("equal"); sol.axis("off")
sol.set_title("Giriş (4×4) — kalın sayılar seçilen en büyük değerler")

for i in range(2):
    for j in range(2):
        sag.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=renkler[i * 2 + j],
                                edgecolor="gray"))
        sag.text(j + 0.5, 1 - i + 0.5, f"{tf_cikis[i, j]:g}", ha="center",
                 va="center", fontsize=16, fontweight="bold")
sag.set_xlim(0, 2); sag.set_ylim(-1, 3); sag.set_aspect("equal"); sag.axis("off")
sag.set_title("Çıkış (2×2)")
fig.suptitle("2×2 Max Pooling, stride = 2", y=1.04)
yol = os.path.join(CIKTI_KLASORU, "bolum4_pooling.png")
fig.savefig(yol, dpi=150, bbox_inches="tight")
plt.close(fig)
yaz(f"Pooling şekli kaydedildi: {yol}")


x = tf.constant([-5., -2., -1., 0., 2., 4., 6.])
relu = tf.keras.layers.ReLU()
y = relu(x)

yaz("\n=== Bölüm 5A: ReLU ===")
yaz("Giriş:", x.numpy())
yaz("ReLU çıkışı:", y.numpy())
elle_relu = np.maximum(0.0, x.numpy())
yaz("max(0, x) ile elle:", elle_relu)
yaz("Aynı mı?:", bool(np.array_equal(elle_relu, y.numpy())))

eksen_x = np.linspace(-6, 7, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(eksen_x, np.maximum(0, eksen_x), label="ReLU: max(0, x)", lw=2)
ax.plot(eksen_x, eksen_x, "--", color="gray", lw=1, label="Doğrusal: y = x")
ax.scatter(x.numpy(), y.numpy(), color="crimson", zorder=3, label="Ödevdeki noktalar")
for xi, yi in zip(x.numpy(), y.numpy()):
    ax.annotate(f"{xi:g}→{yi:g}", (xi, yi), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8)
ax.axhline(0, color="black", lw=0.5); ax.axvline(0, color="black", lw=0.5)
ax.set_xlabel("x"); ax.set_ylabel("f(x)"); ax.set_title("ReLU fonksiyonu")
ax.legend(); ax.grid(alpha=0.3)
yol = os.path.join(CIKTI_KLASORU, "bolum5_relu.png")
fig.savefig(yol, dpi=150, bbox_inches="tight")
plt.close(fig)
yaz(f"ReLU grafiği kaydedildi: {yol}")


logits = tf.constant([[2.0, 1.0, 0.1]])
softmax = tf.keras.layers.Softmax()
probabilities = softmax(logits)

yaz("\n=== Bölüm 5B: Softmax ===")
yaz("Softmax sonuçları:")
yaz(probabilities.numpy())
yaz("Olasılıkların toplamı:")
yaz(tf.reduce_sum(probabilities).numpy())

z = logits.numpy()[0]
usteller = np.exp(z)
elle_softmax = usteller / usteller.sum()
yaz("\nElle hesaplama:")
yaz("  exp(z) =", np.round(usteller, 4), " toplam =", round(float(usteller.sum()), 4))
yaz("  exp(z) / toplam =", np.round(elle_softmax, 6))
yaz("  TensorFlow ile aynı mı? (1e-6):",
    bool(np.allclose(elle_softmax, probabilities.numpy()[0], atol=1e-6)))
tahmin = int(np.argmax(probabilities.numpy()[0]))
yaz(f"  En yüksek olasılık: sınıf {tahmin} (olasılık {elle_softmax[tahmin]:.4f})")


kaydirilmis = tf.keras.layers.Softmax()(logits + 100.0).numpy()[0]
yaz("  Skorlara +100 eklenince softmax:", np.round(kaydirilmis, 6),
    "-> değişmedi mi?:", bool(np.allclose(kaydirilmis, elle_softmax, atol=1e-6)))

fig, (s1, s2) = plt.subplots(1, 2, figsize=(9, 3.6))
siniflar = ["Sınıf 0", "Sınıf 1", "Sınıf 2"]
s1.bar(siniflar, z, color="#7f9cc0")
s1.set_title("Ham skorlar (logit)")
for i, v in enumerate(z):
    s1.text(i, v + 0.03, f"{v:g}", ha="center")
s2.bar(siniflar, elle_softmax, color=["#c0504d" if i == tahmin else "#bbbbbb"
                                       for i in range(3)])
s2.set_title("Softmax olasılıkları (toplam = 1)")
s2.set_ylim(0, 1)
for i, v in enumerate(elle_softmax):
    s2.text(i, v + 0.02, f"{v:.3f}", ha="center")
fig.tight_layout()
yol = os.path.join(CIKTI_KLASORU, "bolum5_softmax.png")
fig.savefig(yol, dpi=150, bbox_inches="tight")
plt.close(fig)
yaz(f"Softmax grafiği kaydedildi: {yol}")

metin_yolu = os.path.join(CIKTI_KLASORU, "bolum4_5_sonuclar.txt")
with open(metin_yolu, "w", encoding="utf-8") as f:
    f.write("\n".join(rapor_satirlari) + "\n")
print(f"Sonuçlar kaydedildi: {metin_yolu}")
