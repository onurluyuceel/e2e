import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


# 1. Excel dosyanızı okuyun
dosya_yolu = 'gkb_hrb.xlsx'
df = pd.read_excel(dosya_yolu)

# 2. Frekans tablosunu oluşturun ve X ekseninin düzgün olması için miktara göre sıralayın
frekans_tablosu = df['ORDER_MIKTAR'].value_counts().reset_index()
frekans_tablosu.columns = ['ORDER_MIKTAR', 'Frekans (Adet)']
frekans_tablosu = frekans_tablosu.sort_values(by='ORDER_MIKTAR')


# 3. Çizgi Grafiğini Çizdirin
plt.figure(figsize=(12, 6))


# X ekseninde miktar, Y ekseninde frekans olacak şekilde çizgi grafiği (plot)
plt.plot(frekans_tablosu['ORDER_MIKTAR'], frekans_tablosu['Frekans (Adet)'],
        color='coral', marker='o', linestyle='-', linewidth=2, markersize=4)


plt.title('Sipariş Miktarı Dağılımı (Çizgi Grafiği)')
plt.xlabel('Sipariş Miktarı (ORDER_MIKTAR)')
plt.ylabel('Frekans (Kaç Kere Sipariş Edildiği)')


# Arka plan çizgilerini ekleyelim
plt.grid(linestyle='--', alpha=0.75)


plt.show()


# 2. MAINPART sütunundaki değerleri sayın ve en yüksek 10 tanesini alın
mainpart_frekans = df['MAINPART'].value_counts().head(10).reset_index()


# 3. Sütun isimlerini daha anlaşılır olacak şekilde düzenleyin
mainpart_frekans.columns = ['MAINPART', 'Frekans (Adet)']


# 4. Oluşturduğunuz tabloyu ekrana yazdırın
print("En Çok Tekrar Eden İlk 10 MAINPART:")
print("-" * 35) # Görsel olarak ayırmak için bir çizgi çekiyoruz
print(mainpart_frekans)


# 2. Tarih sütunlarını datetime formatına çevirin
df['İLK_BARKOD_TARİH'] = pd.to_datetime(df['İLK_BARKOD_TARİH'])
df['PO_CREATIONDATE'] = pd.to_datetime(df['PO_CREATIONDATE'])


# 3. İki tarih arasındaki gün farkını ana tabloya (df) sütun EKLEMEDEN hesaplayın
fark_gun = (df['İLK_BARKOD_TARİH'] - df['PO_CREATIONDATE']).dt.days


# 4. DAĞILIM GRAFİĞİ (Histogram) Çizimi
plt.figure(figsize=(12, 6))


# fark_gun.dropna() kullanarak boş (tarihi girilmemiş) satırların hata vermesini engelliyoruz
# bins=30 ile veriyi 30 farklı zaman aralığına bölüyoruz, isterseniz bu sayıyı değiştirebilirsiniz
plt.hist(fark_gun.dropna(), bins=30, color='mediumpurple', edgecolor='black')


plt.title('Lead Time (Bekleme Süresi) Dağılım Grafiği')
plt.xlabel('Bekleme Süresi (Gün)')
plt.ylabel('Frekans (Sipariş Sayısı)')
plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.show()


# 2. VENDORFINAL sütunundaki değerleri sayıp frekans tablosunu oluşturun
vendor_frekans = df['VENDORFINAL'].value_counts().reset_index()


# 3. Sütun isimlerini daha anlaşılır olacak şekilde düzenleyin
vendor_frekans.columns = ['VENDORFINAL (Tedarikçi Kodu)', 'Frekans (Adet)']


# 4. Oluşturduğunuz tabloyu ekrana yazdırın
print("Tedarikçi (VENDORFINAL) Dağılımı:")
print("-" * 40)
print(vendor_frekans)


# 2. VENDORFINAL sütununa göre gruplayıp, ORDER_MIKTAR sütunundaki değerleri toplayın
# .reset_index() ile sonucu düzgün bir tablo formatına getiriyoruz
vendor_toplam_miktar = df.groupby('VENDORFINAL')['ORDER_MIKTAR'].sum().reset_index()


# 3. Sonucu en yüksek sipariş miktarından en düşüğe doğru sıralayın
vendor_toplam_miktar = vendor_toplam_miktar.sort_values(by='ORDER_MIKTAR', ascending=False).reset_index(drop=True)


# 4. Sütun isimlerini daha anlaşılır olacak şekilde düzenleyin
vendor_toplam_miktar.columns = ['VENDORFINAL (Tedarikçi Kodu)', 'Toplam Sipariş Miktarı']


# 5. Tabloyu ekrana yazdırın
print("Tedarikçi Bazlı Toplam Sipariş Miktarları:")
print("-" * 50)
print(vendor_toplam_miktar)


# 2. Sizin verdiğiniz malzeme sözlüğü
material_map = {
   '6AL-4V (AB-1)': 'Titanyum',
   'TI-6AL-4V': 'Titanyum',
   '321': 'Çelik',
   '15-5PH': 'Çelik',
   '17-HPH': 'Çelik',
   'AISI 301': 'Çelik',
   'AISI321': 'Çelik',
   'CRES 304': 'Çelik',
   'PH13-8MO': 'Çelik',
   'AISI 41410': 'Çelik',
   'INCONEL 625': 'Nikel-krom',
   'INCONEL 718': 'Nikel-krom',
   'C63000': 'Nikel-Alüminyum-Bronze',
   'C64200': 'Alüminyum-Bronze',
   '2050': 'Alüminyum',
   '6061': 'Alüminyum',
   '7050': 'Alüminyum',
   '7075': 'Alüminyum',
   '2024-CLAD': 'Alüminyum',
   '5.1129-1': 'Polytetrafluoroethylene (PTFE)',
   'AERMET 100': 'Çelik',
   '1.7734': 'Çelik',
   'AERO100': 'Çelik',
   '17-7PH': 'Çelik',
   '5052': 'Alüminyum',
   '2024': 'Alüminyum',
   '0.70 - 1.00C': 'Çelik',
   'AISI 304': 'Çelik',
   'SAE 4340': 'Çelik',
   'AISI 321': 'Çelik'
}


# YENİ ADIM: Sözlükteki anahtarları eşleşmeye hazır, standart ve temiz bir hale getirelim
# (Hepsini büyük harf yapar ve kelime aralarındaki birden fazla boşluğu tek boşluğa düşürür)
temiz_material_map = {
   ' '.join(str(k).upper().split()): v for k, v in material_map.items()
}


# 3. Tablodaki 'MATERIALTYPE' değerlerinin FREKANSLARINI (hangisinden kaç tane var) alıyoruz
# (dropna=True ile boş olanları saymıyoruz)
malzeme_frekanslari = df['MATERIALTYPE'].value_counts(dropna=True)


# 4. Sözlükte KARŞILIĞI OLMAYANLARI filtreleyin ve sayılarını eşleştirin
karsiligi_olmayanlar = {}  # Bu sefer liste yerine sözlük kullanıyoruz ki sayıları da tutabilelim


for orijinal_malzeme, adet in malzeme_frekanslari.items():
   # Excel'den gelen veriyi de aynı standart temizleme işleminden geçiriyoruz
   temiz_malzeme = ' '.join(str(orijinal_malzeme).upper().split())


   # Temizlenmiş hali, temizlenmiş sözlüğümüzde var mı diye bakıyoruz
   if temiz_malzeme not in temiz_material_map:
       # Eğer yoksa, orijinal halini ve tablodaki sayısını kaydediyoruz
       karsiligi_olmayanlar[orijinal_malzeme] = adet


# 5. Sonuçları ekrana yazdırın
print("Sözlükte (material_map) Karşılığı BULUNMAYAN Değerler ve Sayıları:")
print("-" * 65)


if len(karsiligi_olmayanlar) == 0:
   print("Harika! Tablodaki tüm malzeme tiplerinin sözlükte bir karşılığı var.")
else:
   toplam_eslesmeyen_satir = 0  # Toplamı tutacağımız sayaç


   # Hem adı hem de adedi (kaç tane olduğunu) ekrana yazdırıyoruz
   for eksik, adet in karsiligi_olmayanlar.items():
       print(f"- '{eksik}': {adet} adet")
       toplam_eslesmeyen_satir += adet  # Her bulduğumuz adedi toplama ekliyoruz


   # En alta toplam eşleşmeyen satır sayısını yazdırıyoruz
   print("-" * 65)
   print(f"TOPLAM EŞLEŞMEYEN SATIR SAYISI: {toplam_eslesmeyen_satir}")

# --- EKSİK (BOŞ) VERİ ANALİZİ ---

# 1. Her sütundaki boş değerlerin toplamını hesaplayıp bir tabloya çevirelim
bos_degerler = df.isnull().sum().reset_index()

# 2. Sütun isimlerini anlaşılır hale getirelim
bos_degerler.columns = ['Sütun Adı', 'Boş Değer Sayısı']

# 3. Sadece boş değeri olan sütunları görmek ve en çok boşluk olandan aza doğru sıralamak için:
bos_degerler = bos_degerler[bos_degerler['Boş Değer Sayısı'] > 0]
bos_degerler = bos_degerler.sort_values(by='Boş Değer Sayısı', ascending=False).reset_index(drop=True)

# 4. Sonucu ekrana yazdıralım
print("\nSütunlardaki Boş (Eksik) Veri Sayıları:")
print("-" * 50)
if len(bos_degerler) == 0:
    print("Mükemmel! Veri setinizde hiçbir boş değer bulunmuyor.")
else:
    print(bos_degerler)
print("-" * 50)

# 1. DIMENSIONCODE için eşleştirme sözlüğü
mapping = {
   'EXTRUSION METALLIC': 'EXTRUSION',
   'RECTANGULAR BAR METALLIC': 'PLATE',
   'SHEET METALLIC': 'SHEET',
   'PLATE METALLIC': 'PLATE',
   'FORGING(RAW)': 'PLATE',
   'MESH METALLIC': 'PLATE',
   'STD. FLAT, SEMI FIN PARTS': 'PLATE',
   'ROUND BAR METALLIC': 'ROUND',
   'ROUND TUBE METALLIC': 'ROUND'
}


# 2. Sözlükteki anahtarları temizleyelim (Büyük harf ve tek boşluk)
temiz_mapping = {
   ' '.join(str(k).upper().split()): v for k, v in mapping.items()
}


# 3. Tablodaki 'DIMENSIONCODE' değerlerinin frekanslarını alalım
boyut_frekanslari = df['DIMENSIONCODE'].value_counts(dropna=True)


# 4. Sözlükte karşılığı olmayanları filtrele
karsiligi_olmayanlar = {}


for orijinal_deger, adet in boyut_frekanslari.items():
   # Veriyi temizle
   temiz_deger = ' '.join(str(orijinal_deger).upper().split())


   if temiz_deger not in temiz_mapping:
       karsiligi_olmayanlar[orijinal_deger] = adet


# 5. Sonuçları ekrana yazdırın
print("DIMENSIONCODE Sözlüğünde Karşılığı BULUNMAYAN Değerler ve Sayıları:")
print("-" * 65)


if len(karsiligi_olmayanlar) == 0:
   print("Harika! Tüm DIMENSIONCODE değerleri sözlükte mevcut.")
else:
   toplam_eslesmeyen_satir = 0


   for eksik, adet in karsiligi_olmayanlar.items():
       print(f"- '{eksik}': {adet} adet")
       toplam_eslesmeyen_satir += adet


   print("-" * 65)
   print(f"TOPLAM EŞLEŞMEYEN SATIR SAYISI: {toplam_eslesmeyen_satir}")


# 1. Tarih sütunlarını datetime formatına çevirelim (Hata almamak için errors='coerce' ekledik)
df['PO_CREATIONDATE'] = pd.to_datetime(df['PO_CREATIONDATE'], errors='coerce')
df['İLK_BARKOD_TARİH'] = pd.to_datetime(df['İLK_BARKOD_TARİH'], errors='coerce')


# 2. Eşik tarihimizi belirleyelim
esik_tarih = pd.Timestamp('2024-10-01')


# --- SENARYO 1: 1 Ekim öncesi açılıp, 1 Ekim sonrası tamamlananlar (Sarkan Siparişler) ---
sarkanlar = df[(df['PO_CREATIONDATE'] < esik_tarih) & (df['İLK_BARKOD_TARİH'] > esik_tarih)]


# --- SENARYO 2: 1 Ekim öncesi açılıp, 1 Ekim öncesi tamamlananlar (Eski Kapananlar) ---
eski_kapananlar = df[(df['PO_CREATIONDATE'] < esik_tarih) & (df['İLK_BARKOD_TARİH'] < esik_tarih)]


# --- SENARYO 3: Her iki tarihi de 1 Ekim sonrası olanlar (Yeni Siparişler) ---
yeni_surecler = df[(df['PO_CREATIONDATE'] >= esik_tarih) & (df['İLK_BARKOD_TARİH'] >= esik_tarih)]


# 3. Sonuçları yazdıralım
print("Tarih Bazlı Sipariş Analizi (Eşik: 1 Ekim 2024)")
print("-" * 60)
print(f"1. Ekim Öncesi Açılan / Ekim Sonrası Biten: {len(sarkanlar)} adet")
print(f"2. Ekim Öncesi Açılan / Ekim Öncesi Biten : {len(eski_kapananlar)} adet")
print(f"3. Her Şeyi Ekim Sonrası Başlayan ve Biten : {len(yeni_surecler)} adet")
print("-" * 60)

# Toplam analize giren satır sayısı (Tarihi eksik olanlar analize girmez)
toplam_gecerli = len(sarkanlar) + len(eski_kapananlar) + len(yeni_surecler)
print(f"Analiz Edilen Toplam Geçerli Satır: {toplam_gecerli}")
