"""
kamus_kode.py
=============
Penerjemah kode angka pada dataset Jaya Jaya Institut menjadi label yang dapat
dibaca manusia.

Dataset menyimpan hampir seluruh variabel kategori sebagai bilangan bulat. Angka
9500 pada kolom Course, misalnya, adalah nomor program studi Keperawatan dan
bukan besaran yang bisa dibandingkan dengan angka 33. Tanpa penerjemahan ini
grafik pada notebook, kartu pada dashboard, dan formulir pada prototipe hanya
akan menampilkan deretan angka yang tidak berarti bagi staf akademik.

Sumber acuan kode adalah dokumentasi dataset Predict Students Dropout and
Academic Success pada UCI Machine Learning Repository, dataset yang sama dengan
yang dibagikan Dicoding pada tautan students performance.
"""

# ---------------------------------------------------------------------------
# Program studi
# ---------------------------------------------------------------------------
PROGRAM_STUDI = {
    33: "Teknologi Produksi Biofuel",
    171: "Desain Animasi dan Multimedia",
    8014: "Pekerjaan Sosial (kelas malam)",
    9003: "Agronomi",
    9070: "Desain Komunikasi",
    9085: "Keperawatan Hewan",
    9119: "Teknik Informatika",
    9130: "Manajemen Kuda",
    9147: "Manajemen",
    9238: "Pekerjaan Sosial",
    9254: "Pariwisata",
    9500: "Keperawatan",
    9556: "Kesehatan Gigi dan Mulut",
    9670: "Manajemen Periklanan dan Pemasaran",
    9773: "Jurnalisme dan Komunikasi",
    9853: "Pendidikan Dasar",
    9991: "Manajemen (kelas malam)",
}

# ---------------------------------------------------------------------------
# Jalur pendaftaran
# ---------------------------------------------------------------------------
JALUR_MASUK = {
    1: "Seleksi umum gelombang 1",
    2: "Jalur peraturan 612/93",
    5: "Kuota khusus wilayah Azores",
    7: "Sudah punya gelar perguruan tinggi lain",
    10: "Jalur peraturan 854-B/99",
    15: "Mahasiswa internasional (sarjana)",
    16: "Kuota khusus wilayah Madeira",
    17: "Seleksi umum gelombang 2",
    18: "Seleksi umum gelombang 3",
    26: "Peraturan 533-A/99 butir b2 (kurikulum berbeda)",
    27: "Peraturan 533-A/99 butir b3 (institusi lain)",
    39: "Jalur usia di atas 23 tahun",
    42: "Pindahan",
    43: "Pindah program studi",
    44: "Pemegang diploma spesialisasi teknologi",
    51: "Pindah institusi dan program studi",
    53: "Pemegang diploma siklus pendek",
    57: "Pindah institusi dan program studi (internasional)",
}

# ---------------------------------------------------------------------------
# Status pernikahan
# ---------------------------------------------------------------------------
STATUS_NIKAH = {
    1: "Belum menikah",
    2: "Menikah",
    3: "Duda atau janda",
    4: "Bercerai",
    5: "Hidup bersama tanpa nikah",
    6: "Pisah secara hukum",
}

# ---------------------------------------------------------------------------
# Kualifikasi pendidikan sebelumnya
# ---------------------------------------------------------------------------
KUALIFIKASI_SEBELUMNYA = {
    1: "Pendidikan menengah tuntas",
    2: "Sarjana",
    3: "Diploma perguruan tinggi",
    4: "Magister",
    5: "Doktor",
    6: "Pernah kuliah tanpa lulus",
    9: "Kelas 12 tidak tuntas",
    10: "Kelas 11 tidak tuntas",
    12: "Setara kelas 11 jalur lain",
    14: "Kelas 10 tuntas",
    15: "Kelas 10 tidak tuntas",
    19: "Pendidikan dasar siklus 3",
    38: "Pendidikan dasar siklus 2",
    39: "Kursus spesialisasi teknologi",
    40: "Sarjana siklus pertama",
    42: "Kursus teknis tinggi kejuruan",
    43: "Magister siklus kedua",
}

# ---------------------------------------------------------------------------
# Pengelompokan kualifikasi orang tua
# ---------------------------------------------------------------------------
# Kode kualifikasi orang tua sangat rinci (34 kode berbeda) sehingga tidak
# praktis ditampilkan satu per satu pada formulir. Kode dikelompokkan menjadi
# empat jenjang yang bermakna bagi staf akademik.
JENJANG_ORANG_TUA = {
    "Perguruan tinggi": [2, 3, 4, 5, 6, 20, 25, 31, 33, 40, 41, 42, 43, 44],
    "Menengah atas": [1, 9, 10, 12, 13, 14, 18, 22, 27, 39],
    "Pendidikan dasar": [11, 19, 26, 29, 30, 37, 38],
    "Tidak bersekolah atau tidak diketahui": [34, 35, 36],
}

# ---------------------------------------------------------------------------
# Pengelompokan pekerjaan orang tua
# ---------------------------------------------------------------------------
# Kode pekerjaan memakai dua tingkat kedalaman: kode satu digit untuk kelompok
# besar dan kode tiga digit untuk rinciannya. Fungsi di bawah menyatukan
# keduanya memakai aturan digit ratusan, bukan daftar manual, agar kode rinci
# yang belum pernah muncul pun tetap tertangani.
KELOMPOK_PEKERJAAN = {
    0: "Pelajar",
    1: "Direktur dan manajer",
    2: "Profesional dan ilmuwan",
    3: "Teknisi tingkat menengah",
    4: "Staf administrasi",
    5: "Layanan personal dan penjualan",
    6: "Pertanian, perikanan, kehutanan",
    7: "Pekerja industri dan konstruksi",
    8: "Operator mesin dan perakitan",
    9: "Pekerja tanpa keahlian khusus",
    10: "Angkatan bersenjata",
}


def kelompok_pekerjaan(kode) -> str:
    """Menerjemahkan kode pekerjaan orang tua menjadi nama kelompok besarnya."""
    try:
        kode = int(kode)
    except (TypeError, ValueError):
        return "Tidak diketahui"
    if kode == 90:
        return "Situasi lain"
    if kode == 99:
        return "Tidak diketahui"
    if kode in KELOMPOK_PEKERJAAN:
        return KELOMPOK_PEKERJAAN[kode]
    if 100 <= kode <= 109:
        # Kode 101 sampai 103 adalah rincian profesi angkatan bersenjata.
        return KELOMPOK_PEKERJAAN[10]
    if kode >= 110:
        # Pada kode tiga digit, digit puluhan menunjukkan kelompok besarnya.
        # Contoh: 171 sampai 175 adalah rincian kelompok 7, pekerja industri
        # dan konstruksi, sedangkan 121 sampai 125 adalah rincian kelompok 2.
        return KELOMPOK_PEKERJAAN.get((kode // 10) % 10, "Situasi lain")
    return "Situasi lain"


def jenjang_orang_tua(kode) -> str:
    """Menerjemahkan kode kualifikasi orang tua menjadi jenjang pendidikannya."""
    try:
        kode = int(kode)
    except (TypeError, ValueError):
        return "Tidak bersekolah atau tidak diketahui"
    for jenjang, daftar in JENJANG_ORANG_TUA.items():
        if kode in daftar:
            return jenjang
    return "Tidak bersekolah atau tidak diketahui"


# ---------------------------------------------------------------------------
# Label untuk kolom biner
# ---------------------------------------------------------------------------
LABEL_BINER = {
    "Daytime_evening_attendance": {1: "Kelas siang", 0: "Kelas malam"},
    "Gender": {1: "Laki laki", 0: "Perempuan"},
    "Displaced": {1: "Merantau", 0: "Tinggal di daerah asal"},
    "Debtor": {1: "Punya tunggakan", 0: "Tanpa tunggakan"},
    "Tuition_fees_up_to_date": {1: "Pembayaran lancar", 0: "Pembayaran tertunggak"},
    "Scholarship_holder": {1: "Penerima beasiswa", 0: "Tanpa beasiswa"},
    "International": {1: "Mahasiswa internasional", 0: "Mahasiswa domestik"},
    "Educational_special_needs": {1: "Berkebutuhan khusus", 0: "Tanpa kebutuhan khusus"},
}

# Nama kolom dalam bahasa Indonesia, dipakai pada label grafik dan tabel.
NAMA_KOLOM = {
    "Marital_status": "Status pernikahan",
    "Application_mode": "Jalur masuk",
    "Application_order": "Urutan pilihan program studi",
    "Course": "Program studi",
    "Daytime_evening_attendance": "Waktu kuliah",
    "Previous_qualification": "Kualifikasi sebelumnya",
    "Previous_qualification_grade": "Nilai kualifikasi sebelumnya",
    "Nacionality": "Kewarganegaraan",
    "Mothers_qualification": "Pendidikan ibu",
    "Fathers_qualification": "Pendidikan ayah",
    "Mothers_occupation": "Pekerjaan ibu",
    "Fathers_occupation": "Pekerjaan ayah",
    "Admission_grade": "Nilai ujian masuk",
    "Displaced": "Status merantau",
    "Educational_special_needs": "Kebutuhan khusus",
    "Debtor": "Status tunggakan",
    "Tuition_fees_up_to_date": "Status pembayaran UKT",
    "Gender": "Jenis kelamin",
    "Scholarship_holder": "Status beasiswa",
    "Age_at_enrollment": "Usia saat mendaftar",
    "International": "Status internasional",
    "Curricular_units_1st_sem_credited": "MK diakui semester 1",
    "Curricular_units_1st_sem_enrolled": "MK diambil semester 1",
    "Curricular_units_1st_sem_evaluations": "Evaluasi semester 1",
    "Curricular_units_1st_sem_approved": "MK lulus semester 1",
    "Curricular_units_1st_sem_grade": "Nilai rata rata semester 1",
    "Curricular_units_1st_sem_without_evaluations": "MK tanpa evaluasi semester 1",
    "Curricular_units_2nd_sem_credited": "MK diakui semester 2",
    "Curricular_units_2nd_sem_enrolled": "MK diambil semester 2",
    "Curricular_units_2nd_sem_evaluations": "Evaluasi semester 2",
    "Curricular_units_2nd_sem_approved": "MK lulus semester 2",
    "Curricular_units_2nd_sem_grade": "Nilai rata rata semester 2",
    "Curricular_units_2nd_sem_without_evaluations": "MK tanpa evaluasi semester 2",
    "Unemployment_rate": "Tingkat pengangguran",
    "Inflation_rate": "Tingkat inflasi",
    "GDP": "Pertumbuhan ekonomi",
    "selisih_adaptasi": "Selisih nilai masuk dan kualifikasi sebelumnya",
    "jeda_masuk": "Jeda tahun sebelum kuliah",
    "dukungan_finansial": "Skor dukungan finansial",
    "pilihan_pertama": "Diterima di pilihan pertama",
    "beban_sks_th1": "Total MK diambil tahun 1",
    "sks_lulus_th1": "Total MK lulus tahun 1",
    "daya_selesai_th1": "Daya selesai tahun 1",
    "daya_selesai_sem1": "Daya selesai semester 1",
    "daya_selesai_sem2": "Daya selesai semester 2",
    "efisiensi_ujian": "Efisiensi evaluasi",
    "nilai_tertimbang": "Nilai tertimbang tahun 1",
    "laju_nilai": "Laju perubahan nilai antar semester",
    "mk_tanpa_evaluasi": "MK tanpa evaluasi tahun 1",
    "sks_diakui": "MK hasil alih kredit",
    "tanpa_beban_studi": "Tidak mengambil MK sama sekali",
}


def nama(kolom: str) -> str:
    """Mengembalikan nama Indonesia sebuah kolom, atau nama aslinya bila tidak ada."""
    return NAMA_KOLOM.get(kolom, kolom)
