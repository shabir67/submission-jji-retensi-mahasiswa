"""
fitur_akademik.py
=================
Definisi kolom dan rekayasa fitur untuk proyek peringatan dini retensi mahasiswa
Jaya Jaya Institut.

Berkas ini menjadi satu satunya sumber definisi fitur. Ia dipakai oleh tiga
tempat sekaligus:

1. ``notebook.ipynb`` ketika melatih model,
2. ``app.py`` ketika melakukan inference pada prototipe Streamlit,
3. skrip di folder ``build`` ketika menyiapkan tabel untuk dashboard Metabase.

Karena fungsi ``tambah_fitur`` dipasang di dalam Pipeline scikit-learn lewat
FunctionTransformer, berkas ini wajib ikut ter-deploy bersama app.py. Joblib
menyimpan referensi fungsi berdasarkan lokasi modulnya, bukan isi fungsinya.

Konsep dua horizon
------------------
Proyek ini melatih dua model untuk pertanyaan yang berbeda waktunya.

* Horizon "gerbang" menjawab pertanyaan pada saat mahasiswa baru diterima,
  ketika institusi belum memiliki satu pun nilai kuliah. Model ini hanya boleh
  memakai data pendaftaran, latar belakang keluarga, dan status pembayaran awal.
* Horizon "tahun_pertama" menjawab pertanyaan setelah dua semester pertama
  selesai, ketika catatan akademik sudah tersedia.

Pemisahan ini penting karena memakai nilai semester dua untuk menilai mahasiswa
yang baru mendaftar adalah kebocoran waktu: informasinya belum ada ketika
keputusan intervensi perlu diambil.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Kolom mentah dataset
# ---------------------------------------------------------------------------

# Kolom yang sudah terisi sejak mahasiswa diterima.
KOLOM_GERBANG = [
    "Marital_status", "Application_mode", "Application_order", "Course",
    "Daytime_evening_attendance", "Previous_qualification", "Previous_qualification_grade",
    "Nacionality", "Mothers_qualification", "Fathers_qualification",
    "Mothers_occupation", "Fathers_occupation", "Admission_grade", "Displaced",
    "Educational_special_needs", "Debtor", "Tuition_fees_up_to_date", "Gender",
    "Scholarship_holder", "Age_at_enrollment", "International",
    "Unemployment_rate", "Inflation_rate", "GDP",
]

# Kolom catatan akademik dua semester pertama, baru terisi di akhir tahun pertama.
KOLOM_AKADEMIK = [
    "Curricular_units_1st_sem_credited", "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations", "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade", "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited", "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations", "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade", "Curricular_units_2nd_sem_without_evaluations",
]

KOLOM_MENTAH = KOLOM_GERBANG + KOLOM_AKADEMIK

# ---------------------------------------------------------------------------
# Fitur turunan
# ---------------------------------------------------------------------------

# Turunan yang hanya butuh data pendaftaran, tersedia di kedua horizon.
TURUNAN_GERBANG = [
    "selisih_adaptasi", "jeda_masuk", "dukungan_finansial", "pilihan_pertama",
]

# Turunan yang butuh catatan akademik, hanya tersedia di horizon tahun pertama.
TURUNAN_AKADEMIK = [
    "beban_sks_th1", "sks_lulus_th1", "daya_selesai_th1", "daya_selesai_sem1",
    "daya_selesai_sem2", "efisiensi_ujian", "nilai_tertimbang", "laju_nilai",
    "mk_tanpa_evaluasi", "sks_diakui", "tanpa_beban_studi",
]


def _bagi_aman(pembilang, penyebut):
    """Membagi dua deret dengan penyebut nol dipetakan menjadi nol, bukan tak hingga.

    Secara bisnis pilihan ini benar: mahasiswa yang tidak mengambil satu pun mata
    kuliah memang tidak menyelesaikan apa pun, sehingga rasionya nol dan bukan
    nilai hilang yang perlu ditebak.
    """
    pembilang = pd.Series(np.asarray(pembilang, dtype=float))
    penyebut = pd.Series(np.asarray(penyebut, dtype=float)).replace(0, np.nan)
    return (pembilang / penyebut).fillna(0.0).to_numpy()


def tambah_fitur(data: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan fitur turunan pada salinan data, tanpa mengubah data asli.

    Fungsi menyesuaikan diri dengan kolom yang tersedia. Bila kolom akademik tidak
    ada, seperti pada horizon gerbang, bagian akademik dilewati sehingga fungsi
    yang sama dapat dipakai oleh kedua model.

    Alasan tiap fitur
    -----------------
    * ``selisih_adaptasi``  : jarak antara nilai ujian masuk dan nilai kualifikasi
      sebelumnya. Nilai negatif besar berarti mahasiswa masuk dengan performa yang
      sedang menurun dibanding jenjang sebelumnya.
    * ``jeda_masuk``        : lama tahun tertunda sebelum kuliah, dihitung dari usia
      18 tahun. Mahasiswa yang menunda lama umumnya sudah memikul tanggung jawab lain.
    * ``dukungan_finansial``: skor gabungan tiga kondisi keuangan menjadi satu sumbu
      tunggal, dari kurang mendukung sampai mendukung penuh.
    * ``pilihan_pertama``   : penanda program studi yang dipilih di urutan teratas.
      Mahasiswa yang terdampar di pilihan kesekian punya keterikatan lebih rendah.
    * ``daya_selesai_*``    : rasio mata kuliah lulus terhadap yang diambil. Jumlah
      mentah tidak adil dibandingkan antar mahasiswa karena beban studinya berbeda.
    * ``efisiensi_ujian``   : berapa banyak evaluasi yang benar benar berbuah kelulusan.
      Mahasiswa yang ikut banyak ujian tetapi sedikit lulus sedang kesulitan.
    * ``nilai_tertimbang``  : rata rata nilai dua semester dengan bobot jumlah mata
      kuliah lulus, sehingga semester berbeban besar berpengaruh lebih besar.
    * ``laju_nilai``        : perubahan relatif nilai semester dua terhadap semester
      satu. Bentuk relatif dipakai agar penurunan pada mahasiswa bernilai rendah
      terbaca lebih tajam daripada penurunan sebesar sama pada mahasiswa bernilai tinggi.
    * ``mk_tanpa_evaluasi`` : mata kuliah yang diambil tetapi tidak pernah dievaluasi,
      pengganti terdekat untuk ketidakhadiran karena dataset tidak mencatat absensi.
    * ``sks_diakui``        : mata kuliah hasil alih kredit, penanda mahasiswa pindahan.
    * ``tanpa_beban_studi`` : penanda mahasiswa yang tidak mengambil mata kuliah sama
      sekali sepanjang tahun pertama, kelompok kecil yang perilakunya ekstrem.
    """
    d = data.copy()

    # --- Bagian gerbang, selalu dihitung ---
    d["selisih_adaptasi"] = d["Admission_grade"] - d["Previous_qualification_grade"]
    d["jeda_masuk"] = (d["Age_at_enrollment"] - 18).clip(lower=0)
    d["dukungan_finansial"] = (
        d["Scholarship_holder"] + d["Tuition_fees_up_to_date"] - d["Debtor"]
    )
    d["pilihan_pertama"] = (d["Application_order"] <= 1).astype(int)

    # --- Bagian akademik, hanya bila kolomnya tersedia ---
    if "Curricular_units_1st_sem_enrolled" not in d.columns:
        return d

    diambil_1 = d["Curricular_units_1st_sem_enrolled"]
    diambil_2 = d["Curricular_units_2nd_sem_enrolled"]
    lulus_1 = d["Curricular_units_1st_sem_approved"]
    lulus_2 = d["Curricular_units_2nd_sem_approved"]
    nilai_1 = d["Curricular_units_1st_sem_grade"]
    nilai_2 = d["Curricular_units_2nd_sem_grade"]

    d["beban_sks_th1"] = diambil_1 + diambil_2
    d["sks_lulus_th1"] = lulus_1 + lulus_2
    d["daya_selesai_th1"] = _bagi_aman(d["sks_lulus_th1"], d["beban_sks_th1"])
    d["daya_selesai_sem1"] = _bagi_aman(lulus_1, diambil_1)
    d["daya_selesai_sem2"] = _bagi_aman(lulus_2, diambil_2)
    d["efisiensi_ujian"] = _bagi_aman(
        d["sks_lulus_th1"],
        d["Curricular_units_1st_sem_evaluations"] + d["Curricular_units_2nd_sem_evaluations"],
    )
    d["nilai_tertimbang"] = _bagi_aman(nilai_1 * lulus_1 + nilai_2 * lulus_2, d["sks_lulus_th1"])
    d["laju_nilai"] = _bagi_aman(nilai_2 - nilai_1, nilai_1)
    d["mk_tanpa_evaluasi"] = (
        d["Curricular_units_1st_sem_without_evaluations"]
        + d["Curricular_units_2nd_sem_without_evaluations"]
    )
    d["sks_diakui"] = (
        d["Curricular_units_1st_sem_credited"] + d["Curricular_units_2nd_sem_credited"]
    )
    d["tanpa_beban_studi"] = (d["beban_sks_th1"] == 0).astype(int)

    return d


# ---------------------------------------------------------------------------
# Pengelompokan kolom untuk preprocessing
# ---------------------------------------------------------------------------

# Kode kategori tanpa urutan alami. Angka 9500 pada kolom Course bukan berarti
# lebih besar daripada 33, ia hanya nomor program studi.
KOLOM_KATEGORI = [
    "Marital_status", "Application_mode", "Course", "Previous_qualification",
    "Nacionality", "Mothers_qualification", "Fathers_qualification",
    "Mothers_occupation", "Fathers_occupation",
]

# Kolom bernilai 0 atau 1 yang sudah siap dipakai apa adanya.
KOLOM_BINER = [
    "Daytime_evening_attendance", "Displaced", "Educational_special_needs", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder", "International",
    "pilihan_pertama", "tanpa_beban_studi",
]


def daftar_fitur(horizon: str):
    """Mengembalikan pasangan (kolom mentah, kolom turunan) untuk horizon tertentu."""
    if horizon == "gerbang":
        return list(KOLOM_GERBANG), list(TURUNAN_GERBANG)
    if horizon == "tahun_pertama":
        return list(KOLOM_MENTAH), list(TURUNAN_GERBANG) + list(TURUNAN_AKADEMIK)
    raise ValueError("Horizon hanya boleh 'gerbang' atau 'tahun_pertama'.")


def bagi_kolom(horizon: str):
    """Memisahkan kolom horizon menjadi tiga kelompok: kategori, biner, dan numerik."""
    mentah, turunan = daftar_fitur(horizon)
    semua = mentah + turunan
    kategori = [k for k in KOLOM_KATEGORI if k in semua]
    biner = [k for k in KOLOM_BINER if k in semua]
    numerik = [k for k in semua if k not in kategori and k not in biner]
    return kategori, biner, numerik


# Pengelompokan tematik atas kolom mentah, dipakai pada analisis kepentingan per
# blok di notebook. Sengaja hanya berisi kolom mentah: fitur turunan dihitung di
# dalam Pipeline dari kolom mentah ini, sehingga mengacak satu blok otomatis ikut
# mengacak seluruh fitur turunan yang bersumber darinya.
BLOK_FITUR = {
    "Akademik tahun pertama": list(KOLOM_AKADEMIK),
    "Finansial": ["Debtor", "Tuition_fees_up_to_date", "Scholarship_holder"],
    "Riwayat pendaftaran": [
        "Application_mode", "Application_order", "Course", "Previous_qualification",
        "Previous_qualification_grade", "Admission_grade", "Daytime_evening_attendance",
    ],
    "Demografi": [
        "Marital_status", "Gender", "Age_at_enrollment", "Nacionality", "International",
        "Displaced", "Educational_special_needs",
    ],
    "Latar keluarga": [
        "Mothers_qualification", "Fathers_qualification",
        "Mothers_occupation", "Fathers_occupation",
    ],
    "Kondisi makroekonomi": ["Unemployment_rate", "Inflation_rate", "GDP"],
}

# Ambang keputusan dan pita risiko hasil analisis biaya pada notebook. Nilai
# dituliskan ulang oleh notebook ke dalam metadata model, konstanta di bawah ini
# hanya cadangan bila metadata tidak terbaca.
PITA_RISIKO = ["Aman", "Prioritas", "Kritis"]


def pita_risiko(probabilitas, ambang: float, batas_kritis: float = 0.65):
    """Memetakan probabilitas menjadi tiga pita tindak lanjut.

    Batas bawah sengaja diikatkan pada ambang keputusan model, bukan pada angka
    bulat, supaya pita di atas "Aman" persis berisi mahasiswa yang memang
    dipanggil model untuk dibimbing. Pita "Kritis" adalah bagian teratas yang
    perlu penanganan paling intensif.
    """
    nilai = np.asarray(probabilitas, dtype=float)
    hasil = np.where(nilai >= batas_kritis, "Kritis",
                     np.where(nilai >= ambang, "Prioritas", "Aman"))
    return pd.Series(hasil, index=getattr(probabilitas, "index", None))
