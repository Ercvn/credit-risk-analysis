# 🏦 Uçtan Uca Kredi Risk Skorlaması ve Limit Tahmini (End-to-End Credit Risk Scoring)

Bu proje, bir bankanın kredi tahsis sürecini yapay zeka ile otomatize etmeyi ve finansal riskleri (temerrüt/batık kredi) minimize etmeyi amaçlayan uçtan uca bir Makine Öğrenmesi (Machine Learning) vaka çalışmasıdır.

Proje, sadece yüksek doğruluk (accuracy) oranlarına ulaşmayı değil; dengesiz veri setleriyle başa çıkmayı, ticari hedeflere göre model optimizasyonu yapmayı ve **Veri Sızıntısı (Data Leakage)** gibi kritik problemleri çözmeyi içerir.

## 🎯 Proje Hedefleri
Proje iki ana makine öğrenmesi görevinden oluşmaktadır:
1. **Sınıflandırma (Risk Skorlaması):** Müşterinin demografik ve finansal geçmişine bakarak kredi borcunu ödeyip ödeyemeyeceğini (Temerrüt Riski) tahmin etmek.
2. **Regresyon (Limit Belirleme):** Kredisi onaylanan "sorunsuz" müşterilere, risk profillerine uygun algoritmik bir kredi limiti (Dolar bazında) tahsis etmek.

## 🛠️ Kullanılan Teknolojiler
* **Veri İşleme ve Temizleme:** `Pandas`, `NumPy`
* **Keşifsel Veri Analizi (EDA):** `Matplotlib`, `Seaborn`
* **Makine Öğrenmesi Algoritmaları:** `Scikit-learn` (Random Forest Classifier & Regressor)

## 📊 Veri Seti
Kaggle'dan alınan [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) kullanılmıştır. Veri seti müşterilerin yaş, gelir, çalışma süresi, ev sahipliği durumu ve geçmiş temerrüt durumlarını içermektedir.

---

## 🚀 Proje Adımları ve Ticari Kazanımlar

### 1. Veri Kalitesi ve Mühendisliği (Data Cleaning & Preprocessing)
* İnsan doğasına aykırı olan ekstrem değerler (144 yaşında olan veya 120 yıldır çalışan müşteriler) tespit edilip veri setinden temizlendi.
* Eksik (Null) veriler, aykırı değerlerin etkisini sıfırlamak adına ortalama (mean) yerine **medyan (median)** ile dolduruldu.
* Makine öğrenmesi algoritmaları için kategorik metin verilerine (örn: Ev Sahibi/Kiracı) `One-Hot Encoding` uygulandı ve kukla değişken tuzağından (Dummy Variable Trap) kaçınıldı.

### 2. Risk Sınıflandırması ve Eşik Değeri Optimizasyonu (Threshold Tuning)
* **Problem:** Dengesiz veri setlerinde (imbalanced data) modelin çoğunluk sınıfına (krediyi ödeyenler) odaklanması ve batık kredileri gözden kaçırması.
* **Aksiyon:** Random Forest Classifier modeli `class_weight='balanced'` parametresi ile eğitildi. Bankanın ticari risklerini sıfıra yaklaştırmak için modelin varsayılan **%50'lik karar eşik değeri (threshold) %30'a çekildi.**
* **Ticari Sonuç:** Bankayı dolandıracak riskli müşterileri yakalama oranımız (**Recall**) %73'ten %78'e yükseltildi. Sistemin "evhamlı" yapısı nedeniyle bazı dürüst müşteriler reddedilse de, bankacılık mantığında *ana para kaybının engellenmesi*, *faiz geliri kaybından* çok daha değerli olduğu için risk yönetimi başarıyla sağlandı.

### 3. Kredi Limiti Tahmini ve "Veri Sızıntısı" (Data Leakage) Çözümü
* **Problem:** Regresyon modeli kurularak müşteriye ne kadar kredi verileceği tahmin edilmek istendi. İlk model $R^2 = 1.00$ gibi kusursuz ve gerçek dışı bir skor üretti.
* **Aksiyon (Analitik Şüphecilik):** Yapılan incelemede, veri seti içindeki `loan_percent_income` (Kredinin Gelire Oranı) sütununun, hedef değişkene giden formülün bir parçası olduğu tespit edildi. Modelin "kopya çekmesine" neden olan bu Veri Sızıntısı (Data Leakage) engellendi ve sorunlu özellik (feature) veri setinden atıldı.
* **Sonuç:** Model gerçek dünya şartlarına uygun hale getirildi ve sadece demografik özelliklere bakarak, kabul edilebilir bir hata payı ($R^2 = 0.26$, MAE = $3885) ile algoritmik limit tahmini yapmayı başardı.

---

## 💻 Nasıl Çalıştırılır?

1. Projeyi bilgisayarınıza klonlayın:
   ```bash
   git clone [https://github.com/Ercvn/credit-risk-analysis.git](https://github.com/Ercvn/credit-risk-analysis.git)

2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn

3. credit_risk_dataset.csv dosyasını ana dizine ekleyin ve Jupyter Notebook (.ipynb) dosyasını çalıştırın.
