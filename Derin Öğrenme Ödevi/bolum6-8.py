import csv
import os
import time

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  


TOHUM = 42
EPOCH = 5
BATCH = 64
DOGRULAMA_ORANI = 0.1
ILK_FILTRELER = {"Deney 1": 8, "Deney 2": 16}
CIKTI_KLASORU = "ciktilar"
SINIF_ADLARI = ["Tişört/Üst", "Pantolon", "Kazak", "Elbise", "Mont",
                "Sandalet", "Gömlek", "Spor ayakkabı", "Çanta", "Bilekte bot"]

os.makedirs(CIKTI_KLASORU, exist_ok=True)
gunluk = []


def yaz(*parcalar):
    satir = " ".join(str(p) for p in parcalar)
    print(satir)
    gunluk.append(satir)


def kaydet(fig, ad):
    yol = os.path.join(CIKTI_KLASORU, ad)
    fig.savefig(yol, dpi=150, bbox_inches="tight")
    plt.close(fig)
    yaz(f"Kaydedildi: {yol}")



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

yaz("=== Bölüm 6A: Veri ===")
yaz(f"Ham piksel aralığı: {x_train.min()} - {x_train.max()}, tür: {x_train.dtype}")

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train[..., np.newaxis]
x_test = x_test[..., np.newaxis]

yaz("Eğitim verisi:", x_train.shape)
yaz("Test verisi:", x_test.shape)
yaz(f"Normalizasyon sonrası aralık: {x_train.min():.1f} - {x_train.max():.1f}")
yaz("Eğitim kümesindeki sınıf sayıları:", np.bincount(y_train).tolist())
dogrulama_adedi = int(len(x_train) * DOGRULAMA_ORANI)
yaz(f"validation_split={DOGRULAMA_ORANI}: {len(x_train) - dogrulama_adedi} eğitim, "
    f"{dogrulama_adedi} doğrulama örneği")


fig, eksenler = plt.subplots(2, 5, figsize=(10, 4.6))
for sinif, eksen in enumerate(eksenler.flat):
    indeks = int(np.where(y_train == sinif)[0][0])
    eksen.imshow(x_train[indeks, :, :, 0], cmap="gray")
    eksen.set_title(f"{sinif}: {SINIF_ADLARI[sinif]}", fontsize=9)
    eksen.axis("off")
fig.suptitle("Fashion-MNIST: her sınıftan bir eğitim örneği (28×28, gri seviye)")
kaydet(fig, "bolum6_ornek_goruntuler.png")



def model_olustur(ilk_filtre):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(ilk_filtre, (3, 3), activation="relu"),   
        tf.keras.layers.MaxPooling2D((2, 2)),                           
        tf.keras.layers.Conv2D(16, (3, 3), activation="relu"),          
        tf.keras.layers.MaxPooling2D((2, 2)),                          
        tf.keras.layers.Flatten(),                                      
        tf.keras.layers.Dense(64, activation="relu"),                   
        tf.keras.layers.Dense(10, activation="softmax"),                
    ])


def katman_tablosu(model):
    """Her katmanın çıkış şeklini ve parametre sayısını tablo olarak döndürür."""
    satirlar = []
    ara = tf.zeros((1, 28, 28, 1))         
    for katman in model.layers:
        ara = katman(ara)
        satirlar.append((katman.name, katman.__class__.__name__,
                         tuple(ara.shape[1:]), katman.count_params()))
    return satirlar



sonuclar = {}
gecmisler = {}
modeller = {}

for deney_adi, ilk_filtre in ILK_FILTRELER.items():
    yaz(f"\n=== {deney_adi}: ilk konvolüsyonda {ilk_filtre} filtre ===")
    tf.keras.utils.set_random_seed(TOHUM)   
    model = model_olustur(ilk_filtre)

    ozet = []
    model.summary(print_fn=lambda s, **kw: ozet.append(s))
    yaz("\n".join(ozet))
    yaz("Katman | Sınıf | Çıkış şekli (batch hariç) | Parametre")
    for ad, sinif, sekil, param in katman_tablosu(model):
        yaz(f"{ad} | {sinif} | {sekil} | {param}")

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    baslangic = time.perf_counter()
    history = model.fit(x_train, y_train,
                        epochs=EPOCH,
                        batch_size=BATCH,
                        validation_split=DOGRULAMA_ORANI,
                        verbose=2)
    egitim_suresi = time.perf_counter() - baslangic

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    h = history.history
    sonuclar[deney_adi] = {
        "ilk_filtre": ilk_filtre,
        "parametre": model.count_params(),
        "egitim_dogrulugu": h["accuracy"][-1],
        "dogrulama_dogrulugu": h["val_accuracy"][-1],
        "test_dogrulugu": test_accuracy,
        "test_kaybi": test_loss,
        "egitim_suresi_sn": egitim_suresi,
        "epoch_basina_sn": egitim_suresi / EPOCH,
    }
    gecmisler[deney_adi] = h
    modeller[deney_adi] = model

    yaz(f"Son epoch eğitim doğruluğu : {h['accuracy'][-1]:.4f}")
    yaz(f"Son epoch doğrulama doğr.  : {h['val_accuracy'][-1]:.4f}")
    yaz(f"Test doğruluğu             : {test_accuracy:.4f}")
    yaz(f"Test kaybı                 : {test_loss:.4f}")
    yaz(f"Eğitim süresi (model.fit)  : {egitim_suresi:.1f} sn")

    
    epochlar = range(1, EPOCH + 1)
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochlar, h["accuracy"], "o-", label="Training")
    plt.plot(epochlar, h["val_accuracy"], "s-", label="Validation")
    plt.title(f"Accuracy — {deney_adi} ({ilk_filtre} filtre)")
    plt.xlabel("Epoch"); plt.xticks(list(epochlar)); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(epochlar, h["loss"], "o-", label="Training")
    plt.plot(epochlar, h["val_loss"], "s-", label="Validation")
    plt.title(f"Loss — {deney_adi} ({ilk_filtre} filtre)")
    plt.xlabel("Epoch"); plt.xticks(list(epochlar)); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    kaydet(fig, f"bolum6_egitim_grafigi_{ilk_filtre}_filtre.png")



ana_model = modeller["Deney 1"]
olasiliklar = ana_model.predict(x_test, verbose=0)
tahminler = np.argmax(olasiliklar, axis=1)

yaz("\n=== Bölüm 7: Softmax çıkışından tahmin ===")
yaz("İlk test görüntüsünün softmax çıktısı:", np.round(olasiliklar[0], 4))
yaz(f"Olasılıkların toplamı: {olasiliklar[0].sum():.4f}")
yaz(f"argmax -> sınıf {tahminler[0]} ({SINIF_ADLARI[tahminler[0]]}), "
    f"gerçek sınıf {y_test[0]} ({SINIF_ADLARI[y_test[0]]})")
yaz(f"argmax ile hesaplanan doğruluk: {np.mean(tahminler == y_test):.4f}")

fig, eksenler = plt.subplots(3, 5, figsize=(11, 7.4))
for i, eksen in enumerate(eksenler.flat):
    eksen.imshow(x_test[i, :, :, 0], cmap="gray")
    dogru = tahminler[i] == y_test[i]
    eksen.set_title(f"Tahmin: {SINIF_ADLARI[tahminler[i]]}\n"
                    f"%{100 * olasiliklar[i].max():.1f} | Gerçek: {SINIF_ADLARI[y_test[i]]}",
                    fontsize=8, color="green" if dogru else "red")
    eksen.axis("off")
fig.suptitle("Deney 1 modeli: ilk 15 test görüntüsünün tahminleri (yeşil = doğru)")
kaydet(fig, "bolum7_ornek_tahminler.png")

karisiklik = np.zeros((10, 10), dtype=int)
for gercek, tahmin in zip(y_test, tahminler):
    karisiklik[gercek, tahmin] += 1
fig, ax = plt.subplots(figsize=(8, 7))
ax.imshow(karisiklik, cmap="Blues")
for (r, c), deger in np.ndenumerate(karisiklik):
    ax.text(c, r, deger, ha="center", va="center", fontsize=7,
            color="white" if deger > karisiklik.max() / 2 else "black")
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xticklabels(SINIF_ADLARI, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(SINIF_ADLARI, fontsize=8)
ax.set_xlabel("Tahmin edilen sınıf"); ax.set_ylabel("Gerçek sınıf")
ax.set_title("Deney 1 modeli: test kümesi karışıklık matrisi")
kaydet(fig, "bolum7_karisiklik_matrisi.png")

sinif_dogrulugu = karisiklik.diagonal() / karisiklik.sum(axis=1)
yaz("Sınıf bazında test doğruluğu:")
for ad, d in sorted(zip(SINIF_ADLARI, sinif_dogrulugu), key=lambda t: t[1]):
    yaz(f"  {ad:<14} {d:.3f}")
karisiklik_kopya = karisiklik.copy()
np.fill_diagonal(karisiklik_kopya, 0)
r, c = np.unravel_index(np.argmax(karisiklik_kopya), karisiklik_kopya.shape)
yaz(f"En sık karıştırılan çift: gerçek '{SINIF_ADLARI[r]}' -> tahmin "
    f"'{SINIF_ADLARI[c]}' ({karisiklik_kopya[r, c]} kez)")


yaz("\n=== Bölüm 8: Karşılaştırma ===")
olcutler = [
    ("Filtre sayısı (ilk katman)", "ilk_filtre", "{:d}"),
    ("Toplam parametre", "parametre", "{:,d}"),
    ("Eğitim doğruluğu (son epoch)", "egitim_dogrulugu", "{:.4f}"),
    ("Doğrulama doğruluğu (son epoch)", "dogrulama_dogrulugu", "{:.4f}"),
    ("Test doğruluğu", "test_dogrulugu", "{:.4f}"),
    ("Test kaybı", "test_kaybi", "{:.4f}"),
    ("Eğitim süresi (sn)", "egitim_suresi_sn", "{:.1f}"),
    ("Epoch başına süre (sn)", "epoch_basina_sn", "{:.1f}"),
]
d1, d2 = sonuclar["Deney 1"], sonuclar["Deney 2"]

md = ["| Ölçüt | Deney 1 | Deney 2 |", "|---|---|---|"]
for baslik, anahtar, bicim in olcutler:
    md.append(f"| {baslik} | {bicim.format(d1[anahtar])} | {bicim.format(d2[anahtar])} |")
yaz("\n".join(md))

fark_test = 100 * (d2["test_dogrulugu"] - d1["test_dogrulugu"])
fark_egitim = 100 * (d2["egitim_dogrulugu"] - d1["egitim_dogrulugu"])
sure_orani = d2["egitim_suresi_sn"] / d1["egitim_suresi_sn"]
ozet = [
    f"Test doğruluğu farkı (Deney 2 - Deney 1): {fark_test:+.2f} yüzde puan",
    f"Eğitim doğruluğu farkı (Deney 2 - Deney 1): {fark_egitim:+.2f} yüzde puan",
    f"Süre oranı (Deney 2 / Deney 1): {sure_orani:.2f}x",
    f"Eğitim-test açığı: Deney 1 {100 * (d1['egitim_dogrulugu'] - d1['test_dogrulugu']):.2f}"
    f" puan, Deney 2 {100 * (d2['egitim_dogrulugu'] - d2['test_dogrulugu']):.2f} puan",
]
for s in ozet:
    yaz(s)

with open(os.path.join(CIKTI_KLASORU, "bolum8_karsilastirma.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n\n" + "\n".join(f"- {s}" for s in ozet) + "\n")
with open(os.path.join(CIKTI_KLASORU, "bolum8_karsilastirma.csv"), "w", newline="",
          encoding="utf-8") as f:
    yazici = csv.writer(f)
    yazici.writerow(["olcut", "deney_1", "deney_2"])
    for baslik, anahtar, _ in olcutler:
        yazici.writerow([baslik, d1[anahtar], d2[anahtar]])

epochlar = range(1, EPOCH + 1)
fig, (s1, s2) = plt.subplots(1, 2, figsize=(11, 4))
for deney_adi, cizgi in [("Deney 1", "-"), ("Deney 2", "--")]:
    h = gecmisler[deney_adi]
    etiket = f"{deney_adi} ({ILK_FILTRELER[deney_adi]} filtre)"
    s1.plot(epochlar, h["val_accuracy"], cizgi, marker="o", label=etiket)
    s2.plot(epochlar, h["val_loss"], cizgi, marker="o", label=etiket)
s1.set_title("Doğrulama doğruluğu"); s2.set_title("Doğrulama kaybı")
for s in (s1, s2):
    s.set_xlabel("Epoch"); s.set_xticks(list(epochlar)); s.legend(); s.grid(alpha=0.3)
fig.tight_layout()
kaydet(fig, "bolum8_deney_karsilastirma.png")

with open(os.path.join(CIKTI_KLASORU, "bolum6_8_sonuclar.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(gunluk) + "\n")
print("Bütün sonuçlar ciktilar/ klasörüne kaydedildi.")
