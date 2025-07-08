# 🌌 Galaxy GAN

Bu proje, Derin Öğrenme tabanlı Generative Adversarial Networks (GAN) kullanılarak yapay galaksi görüntüleri üretmeyi amaçlamaktadır. Eğitimli bir Generator modeli, uzay temalı galaksi benzeri yeni görseller üretir.

## 📌 Proje Özeti

- Girdi olarak yalnızca rastgele latent vektörler (noise) alır.
- Çıktı olarak gri tonlamalı galaksi görselleri üretir.
- TensorFlow 2 ve Keras API kullanılarak baştan sona GAN modeli inşa edilmiştir.
- Eğitim sırasında üretilen örnek görseller `.png` dosyaları olarak kaydedilir.

---

## 🧠 Kullanılan Teknolojiler

- Python
- TensorFlow 2.x (Keras API)
- NumPy, Matplotlib, Tqdm
- DCGAN benzeri yapı (Conv2DTranspose + LeakyReLU + BatchNorm)
- Verisetinin hazırlanması için `tf.data` API

---

## 📁 Dosya Açıklamaları

| Dosya Adı                            | Açıklama |
|--------------------------------------|----------|
| `Generat_Galaxies_using_Gan.py`     | GAN eğitimini başlatan ana dosya. Generator ve Discriminator yapıları, eğitim fonksiyonları ve görsel kaydetme işlevlerini içerir. |
---

