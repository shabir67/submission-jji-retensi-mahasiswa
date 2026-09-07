"""
app.py
======
Prototipe sistem peringatan dini retensi mahasiswa Jaya Jaya Institut.

Aplikasi menyajikan empat ruang kerja yang berbeda kegunaannya:

1. Penilaian mahasiswa  : menilai satu mahasiswa lewat formulir, lengkap dengan
                          faktor yang mendorong dan menahan risikonya.
2. Antrean bimbingan    : menyusun daftar prioritas dari satu berkas CSV sesuai
                          kapasitas konselor yang tersedia.
3. Simulasi kebijakan   : memperkirakan pergeseran risiko satu angkatan bila
                          sebuah kebijakan diterapkan.
4. Tentang model        : ringkasan performa, ambang keputusan, dan batasannya.

Menjalankan secara lokal:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import fitur_akademik as fa
import kamus_kode as kk

# ---------------------------------------------------------------------------
# Konfigurasi halaman dan gaya
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Peringatan Dini Retensi Mahasiswa | Jaya Jaya Institut",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASAR = Path(__file__).parent

# Palet sama dengan palet notebook supaya laporan dan aplikasi terbaca sebagai
# satu kesatuan. Isyarat status memakai warna dan garis tepi, tanpa ikon.
WARNA = {
    "utama": "#0f766e",
    "aksen": "#d97706",
    "kelabu": "#475569",
    "aman": "#15803d",
    "prioritas": "#b45309",
    "kritis": "#b91c1c",
    "garis": "#d8d4cc",
    "tinta": "#141414",
    "permukaan": "#fbfaf8",
}

st.markdown(
    f"""
    <style>
        .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1220px; }}
        h1, h2, h3 {{ color: {WARNA['tinta']}; letter-spacing: -0.01em; }}
        .kepala {{
            border-left: 5px solid {WARNA['utama']};
            padding: 0.1rem 0 0.1rem 0.9rem;
            margin-bottom: 1.1rem;
        }}
        .kepala h1 {{ font-size: 1.65rem; margin: 0 0 0.25rem 0; }}
        .kepala p {{ color: {WARNA['kelabu']}; margin: 0; font-size: 0.92rem; }}
        .kartu {{
            background: {WARNA['permukaan']};
            border: 1px solid {WARNA['garis']};
            border-radius: 8px;
            padding: 1rem 1.1rem;
            margin-bottom: 0.7rem;
        }}
        .kartu-nilai {{ font-size: 2.1rem; font-weight: 700; line-height: 1.1; }}
        .kartu-label {{ font-size: 0.82rem; color: {WARNA['kelabu']};
                        text-transform: uppercase; letter-spacing: 0.04em; }}
        .faktor {{
            padding: 0.55rem 0.8rem;
            margin-bottom: 0.4rem;
            background: #ffffff;
            border: 1px solid {WARNA['garis']};
            border-radius: 6px;
            font-size: 0.9rem;
        }}
        .naik {{ border-left: 4px solid {WARNA['kritis']}; }}
        .turun {{ border-left: 4px solid {WARNA['aman']}; }}
        .catatan {{
            font-size: 0.85rem; color: {WARNA['kelabu']};
            border-top: 1px solid {WARNA['garis']}; padding-top: 0.7rem; margin-top: 1.2rem;
        }}
        div[data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Memuat artefak
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model peringatan dini")
def muat_artefak():
    """Memuat kedua model beserta metadatanya satu kali per sesi."""
    with open(DASAR / "model" / "metadata_model.json", encoding="utf-8") as berkas:
        info = json.load(berkas)
    model = {
        "gerbang": joblib.load(DASAR / "model" / "model_gerbang.joblib"),
        "tahun_pertama": joblib.load(DASAR / "model" / "model_tahun_pertama.joblib"),
    }
    return model, info


@st.cache_data(show_spinner=False)
def muat_contoh():
    """Membaca berkas contoh berisi mahasiswa yang masih aktif."""
    jalur = DASAR / "dataset" / "mahasiswa_aktif.csv"
    if jalur.exists():
        return pd.read_csv(jalur, sep=";")
    return None


MODEL, INFO = muat_artefak()
LABEL_HORIZON = {
    "gerbang": "Gerbang masuk, saat mahasiswa baru diterima",
    "tahun_pertama": "Akhir tahun pertama, setelah dua semester",
}
WARNA_PITA = {"Aman": WARNA["aman"], "Prioritas": WARNA["prioritas"], "Kritis": WARNA["kritis"]}


def ambang(horizon):
    return float(INFO["horizon"][horizon]["ambang_keputusan"])


def batas_kritis():
    return float(INFO["batas_kritis"])


def skor(horizon, data: pd.DataFrame) -> np.ndarray:
    """Menghitung peluang dropout untuk satu atau banyak mahasiswa sekaligus."""
    kolom = INFO["horizon"][horizon]["kolom_mentah"]
    return MODEL[horizon].predict_proba(data[kolom])[:, 1]


def pita(horizon, peluang):
    return fa.pita_risiko(peluang, ambang(horizon), batas_kritis())


def baris_bawaan() -> dict:
    """Nilai bawaan formulir: modus untuk kolom kode, median untuk kolom numerik."""
    return dict(INFO["nilai_bawaan"])


def indeks_bawaan(peta: dict, nilai: int) -> int:
    """Menentukan pilihan awal daftar kelompok yang mewakili mahasiswa lazim.

    Kelompok pendidikan dan pekerjaan orang tua diwakili oleh satu kode. Bila kode
    wakil sebuah kelompok sama dengan nilai bawaan populasi, kelompok itulah yang
    ditampilkan lebih dulu, sehingga formulir kosong benar benar menggambarkan
    mahasiswa lazim dan daftar faktor tidak terisi selisih yang tidak dimaksudkan.
    """
    kunci = list(peta.keys())
    for nomor, nama_kelompok in enumerate(kunci):
        if peta[nama_kelompok] == nilai:
            return nomor
    return 0


# ---------------------------------------------------------------------------
# Bilah sisi
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Jaya Jaya Institut")
    st.caption("Sistem peringatan dini retensi mahasiswa")

    horizon = st.radio(
        "Titik waktu penilaian",
        options=["tahun_pertama", "gerbang"],
        format_func=lambda h: ("Akhir tahun pertama" if h == "tahun_pertama"
                               else "Gerbang masuk"),
        help="Pilih model sesuai data yang sudah tersedia tentang mahasiswa.",
    )
    st.caption(LABEL_HORIZON[horizon])

    rinci = INFO["horizon"][horizon]
    st.markdown("---")
    st.markdown("**Model yang dipakai**")
    st.write(rinci["nama_model"])
    kolom_kiri, kolom_kanan = st.columns(2)
    kolom_kiri.metric("ROC-AUC", f"{rinci['metrik_uji']['roc_auc']:.3f}")
    kolom_kanan.metric("Recall", f"{rinci['metrik_uji']['recall']:.3f}")
    kolom_kiri.metric("Presisi", f"{rinci['metrik_uji']['presisi']:.3f}")
    kolom_kanan.metric("Ambang", f"{rinci['ambang_keputusan']:.2f}")

    st.markdown("---")
    st.caption(
        f"Disusun oleh {INFO['identitas']['nama']} "
        f"(ID Dicoding {INFO['identitas']['id_dicoding']}) untuk proyek akhir kelas "
        "Belajar Penerapan Data Science, Dicoding."
    )

st.markdown(
    f"""
    <div class="kepala">
        <h1>Peringatan Dini Retensi Mahasiswa</h1>
        <p>Menilai risiko berhenti studi pada dua titik waktu, lalu mengubahnya
        menjadi urutan prioritas bimbingan yang sanggup dikerjakan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_nilai, tab_antrean, tab_simulasi, tab_model = st.tabs(
    ["Penilaian mahasiswa", "Antrean bimbingan", "Simulasi kebijakan", "Tentang model"]
)


# ---------------------------------------------------------------------------
# Alat bantu tampilan
# ---------------------------------------------------------------------------
def kartu_angka(kolom, label, nilai, warna=None):
    kolom.markdown(
        f"""
        <div class="kartu">
            <div class="kartu-label">{label}</div>
            <div class="kartu-nilai" style="color:{warna or WARNA['tinta']}">{nilai}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def batang_risiko(nilai, ambang_model):
    """Menggambar garis skala risiko beserta posisi mahasiswa dan letak ambang."""
    posisi = min(max(nilai, 0.0), 1.0) * 100
    letak_ambang = ambang_model * 100
    letak_kritis = batas_kritis() * 100
    st.markdown(
        f"""
        <div style="margin:0.2rem 0 1.1rem 0;">
          <div style="position:relative;height:16px;border-radius:8px;
                      background:linear-gradient(90deg,{WARNA['aman']} 0%,
                      {WARNA['aman']} {letak_ambang}%, {WARNA['prioritas']} {letak_ambang}%,
                      {WARNA['prioritas']} {letak_kritis}%, {WARNA['kritis']} {letak_kritis}%,
                      {WARNA['kritis']} 100%);">
            <div style="position:absolute;left:{posisi}%;top:-6px;width:3px;height:28px;
                        background:{WARNA['tinta']};transform:translateX(-1px);"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:0.75rem;
                      color:{WARNA['kelabu']};margin-top:0.3rem;">
            <span>0,00 aman</span>
            <span>ambang {ambang_model:.2f}</span>
            <span>kritis {batas_kritis():.2f}</span>
            <span>1,00</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def faktor_pendorong(horizon, baris: pd.DataFrame, jumlah=5):
    """Mengukur sumbangan tiap kolom dengan mengembalikannya ke nilai mahasiswa lazim.

    Untuk setiap kolom dibuat satu salinan mahasiswa yang identik kecuali kolom
    tersebut diganti nilai bawaan populasi. Selisih peluang antara mahasiswa asli
    dan salinan itulah sumbangan kolom bersangkutan. Nilai positif berarti kondisi
    mahasiswa pada kolom tersebut menaikkan risikonya dibanding mahasiswa lazim,
    dan nilai negatif berarti menahannya.

    Seluruh salinan dinilai dalam satu panggilan agar tetap ringan dijalankan.
    """
    kolom_model = INFO["horizon"][horizon]["kolom_mentah"]
    bawaan = baris_bawaan()
    dasar = float(skor(horizon, baris)[0])

    kandidat = [k for k in kolom_model if baris[k].iloc[0] != bawaan[k]]
    if not kandidat:
        return dasar, []

    tumpukan = pd.concat([baris] * len(kandidat), ignore_index=True)
    for nomor, kolom in enumerate(kandidat):
        tumpukan.loc[nomor, kolom] = bawaan[kolom]
    peluang_tanpa = skor(horizon, tumpukan)

    sumbangan = sorted(zip(kandidat, dasar - peluang_tanpa), key=lambda x: -abs(x[1]))
    return dasar, [(k, float(v)) for k, v in sumbangan[:jumlah] if abs(v) >= 0.005]


def jelaskan_nilai(kolom, nilai):
    """Menerjemahkan nilai satu kolom menjadi kalimat yang dapat dibaca staf."""
    if kolom in kk.LABEL_BINER:
        return kk.LABEL_BINER[kolom].get(int(nilai), str(nilai))
    if kolom == "Course":
        return kk.PROGRAM_STUDI.get(int(nilai), str(nilai))
    if kolom == "Application_mode":
        return kk.JALUR_MASUK.get(int(nilai), str(nilai))
    if kolom == "Marital_status":
        return kk.STATUS_NIKAH.get(int(nilai), str(nilai))
    if kolom == "Previous_qualification":
        return kk.KUALIFIKASI_SEBELUMNYA.get(int(nilai), str(nilai))
    if kolom in ("Mothers_qualification", "Fathers_qualification"):
        return kk.jenjang_orang_tua(nilai)
    if kolom in ("Mothers_occupation", "Fathers_occupation"):
        return kk.kelompok_pekerjaan(nilai)
    if float(nilai) == int(float(nilai)):
        return str(int(float(nilai)))
    return f"{float(nilai):.2f}"


SARAN_PITA = {
    "Aman": [
        "Cukup dipantau lewat laporan rutin dosen wali tiap akhir semester.",
        "Tidak perlu masuk antrean bimbingan selama tidak ada perubahan status keuangan.",
    ],
    "Prioritas": [
        "Jadwalkan percakapan dengan dosen wali dalam dua pekan ke depan.",
        "Periksa status pembayaran dan tawarkan skema cicilan bila diperlukan.",
        "Tinjau beban studi semester berjalan, pertimbangkan pengurangan mata kuliah.",
    ],
    "Kritis": [
        "Hubungi mahasiswa pekan ini juga, jangan menunggu jadwal bimbingan berikutnya.",
        "Libatkan bagian kemahasiswaan dan bagian keuangan sekaligus dalam satu pertemuan.",
        "Susun kontrak studi dengan target yang dapat diukur pada akhir semester.",
        "Bila tunggakan menjadi penyebab utama, dahulukan penyelesaian keringanan biaya.",
    ],
}


# ---------------------------------------------------------------------------
# Tab 1: penilaian satu mahasiswa
# ---------------------------------------------------------------------------
with tab_nilai:
    st.markdown("#### Menilai satu mahasiswa")
    st.caption(
        "Isi data yang tersedia, lalu tekan tombol di bawah formulir. Kolom yang tidak "
        "diisi memakai nilai mahasiswa lazim pada data historis."
    )

    bawaan = baris_bawaan()
    pilihan = INFO["pilihan"]
    rentang = INFO["rentang"]
    masukan = dict(bawaan)

    with st.form("formulir_mahasiswa"):
        st.markdown("**Identitas dan berkas pendaftaran**")
        baris1 = st.columns(3)
        kode_prodi = baris1[0].selectbox(
            "Program studi", options=[int(k) for k in pilihan["program_studi"]],
            format_func=lambda k: pilihan["program_studi"][str(k)],
            index=list(int(k) for k in pilihan["program_studi"]).index(bawaan["Course"]))
        kode_jalur = baris1[1].selectbox(
            "Jalur masuk", options=[int(k) for k in pilihan["jalur_masuk"]],
            format_func=lambda k: pilihan["jalur_masuk"][str(k)],
            index=list(int(k) for k in pilihan["jalur_masuk"]).index(bawaan["Application_mode"]))
        urutan_pilihan = baris1[2].number_input(
            "Urutan pilihan program studi", min_value=int(rentang["Application_order"]["min"]),
            max_value=int(rentang["Application_order"]["max"]),
            value=int(bawaan["Application_order"]),
            help="1 berarti program studi ini adalah pilihan pertama mahasiswa.")

        baris2 = st.columns(3)
        kode_kualifikasi = baris2[0].selectbox(
            "Kualifikasi pendidikan sebelumnya",
            options=[int(k) for k in pilihan["kualifikasi_sebelumnya"]],
            format_func=lambda k: pilihan["kualifikasi_sebelumnya"][str(k)],
            index=list(int(k) for k in pilihan["kualifikasi_sebelumnya"]).index(
                bawaan["Previous_qualification"]))
        nilai_kualifikasi = baris2[1].slider(
            "Nilai kualifikasi sebelumnya", float(rentang["Previous_qualification_grade"]["min"]),
            float(rentang["Previous_qualification_grade"]["max"]),
            float(bawaan["Previous_qualification_grade"]), step=0.5)
        nilai_masuk = baris2[2].slider(
            "Nilai ujian masuk", float(rentang["Admission_grade"]["min"]),
            float(rentang["Admission_grade"]["max"]), float(bawaan["Admission_grade"]), step=0.5)

        st.markdown("**Data pribadi**")
        baris3 = st.columns(4)
        usia = baris3[0].number_input(
            "Usia saat mendaftar", min_value=int(rentang["Age_at_enrollment"]["min"]),
            max_value=int(rentang["Age_at_enrollment"]["max"]),
            value=int(bawaan["Age_at_enrollment"]))
        jenis_kelamin = baris3[1].selectbox("Jenis kelamin", [0, 1],
                                            format_func=lambda v: kk.LABEL_BINER["Gender"][v],
                                            index=int(bawaan["Gender"]))
        kode_nikah = baris3[2].selectbox(
            "Status pernikahan", options=[int(k) for k in pilihan["status_nikah"]],
            format_func=lambda k: pilihan["status_nikah"][str(k)],
            index=list(int(k) for k in pilihan["status_nikah"]).index(bawaan["Marital_status"]))
        waktu_kuliah = baris3[3].selectbox(
            "Waktu kuliah", [1, 0],
            format_func=lambda v: kk.LABEL_BINER["Daytime_evening_attendance"][v],
            index=0 if bawaan["Daytime_evening_attendance"] == 1 else 1)

        baris4 = st.columns(4)
        merantau = baris4[0].selectbox("Status tempat tinggal", [1, 0],
                                       format_func=lambda v: kk.LABEL_BINER["Displaced"][v],
                                       index=0 if bawaan["Displaced"] == 1 else 1)
        internasional = baris4[1].selectbox(
            "Kewarganegaraan", [0, 1],
            format_func=lambda v: kk.LABEL_BINER["International"][v],
            index=int(bawaan["International"]))
        kebutuhan_khusus = baris4[2].selectbox(
            "Kebutuhan khusus", [0, 1],
            format_func=lambda v: kk.LABEL_BINER["Educational_special_needs"][v],
            index=int(bawaan["Educational_special_needs"]))
        jenjang_ibu = baris4[3].selectbox(
            "Pendidikan ibu", options=list(pilihan["jenjang_ibu"].keys()),
            index=indeks_bawaan(pilihan["jenjang_ibu"], bawaan["Mothers_qualification"]))

        baris5 = st.columns(3)
        jenjang_ayah = baris5[0].selectbox(
            "Pendidikan ayah", options=list(pilihan["jenjang_ayah"].keys()),
            index=indeks_bawaan(pilihan["jenjang_ayah"], bawaan["Fathers_qualification"]))
        pekerjaan_ibu = baris5[1].selectbox(
            "Pekerjaan ibu", options=list(pilihan["pekerjaan_ibu"].keys()),
            index=indeks_bawaan(pilihan["pekerjaan_ibu"], bawaan["Mothers_occupation"]))
        pekerjaan_ayah = baris5[2].selectbox(
            "Pekerjaan ayah", options=list(pilihan["pekerjaan_ayah"].keys()),
            index=indeks_bawaan(pilihan["pekerjaan_ayah"], bawaan["Fathers_occupation"]))

        st.markdown("**Kondisi keuangan**")
        baris6 = st.columns(3)
        ukt_lancar = baris6[0].selectbox(
            "Status pembayaran uang kuliah", [1, 0],
            format_func=lambda v: kk.LABEL_BINER["Tuition_fees_up_to_date"][v],
            index=0 if bawaan["Tuition_fees_up_to_date"] == 1 else 1)
        tunggakan = baris6[1].selectbox("Status tunggakan", [0, 1],
                                        format_func=lambda v: kk.LABEL_BINER["Debtor"][v],
                                        index=int(bawaan["Debtor"]))
        beasiswa = baris6[2].selectbox(
            "Status beasiswa", [0, 1],
            format_func=lambda v: kk.LABEL_BINER["Scholarship_holder"][v],
            index=int(bawaan["Scholarship_holder"]))

        if horizon == "tahun_pertama":
            st.markdown("**Catatan akademik tahun pertama**")
            st.caption("Isi apa adanya sesuai sistem informasi akademik. "
                       "Nilai memakai skala 0 sampai 20 seperti pada dataset asli.")
            akademik = {}
            for nomor_semester, sebutan in [(1, "1st"), (2, "2nd")]:
                st.markdown(f"Semester {nomor_semester}")
                kolom_sem = st.columns(5)
                akademik[f"Curricular_units_{sebutan}_sem_enrolled"] = kolom_sem[0].number_input(
                    "MK diambil", 0, 30, int(bawaan[f"Curricular_units_{sebutan}_sem_enrolled"]),
                    key=f"amb{nomor_semester}")
                akademik[f"Curricular_units_{sebutan}_sem_evaluations"] = kolom_sem[1].number_input(
                    "Evaluasi", 0, 45,
                    int(bawaan[f"Curricular_units_{sebutan}_sem_evaluations"]),
                    key=f"eva{nomor_semester}")
                akademik[f"Curricular_units_{sebutan}_sem_approved"] = kolom_sem[2].number_input(
                    "MK lulus", 0, 30, int(bawaan[f"Curricular_units_{sebutan}_sem_approved"]),
                    key=f"lul{nomor_semester}")
                akademik[f"Curricular_units_{sebutan}_sem_grade"] = kolom_sem[3].slider(
                    "Nilai rata rata", 0.0, 20.0,
                    float(bawaan[f"Curricular_units_{sebutan}_sem_grade"]), step=0.25,
                    key=f"nil{nomor_semester}")
                akademik[f"Curricular_units_{sebutan}_sem_without_evaluations"] = (
                    kolom_sem[4].number_input(
                        "Tanpa evaluasi", 0, 15,
                        int(bawaan[f"Curricular_units_{sebutan}_sem_without_evaluations"]),
                        key=f"tanpa{nomor_semester}"))

        kirim = st.form_submit_button("Hitung risiko mahasiswa", type="primary")

    if kirim:
        masukan.update({
            "Course": kode_prodi,
            "Application_mode": kode_jalur,
            "Application_order": urutan_pilihan,
            "Previous_qualification": kode_kualifikasi,
            "Previous_qualification_grade": nilai_kualifikasi,
            "Admission_grade": nilai_masuk,
            "Age_at_enrollment": usia,
            "Gender": jenis_kelamin,
            "Marital_status": kode_nikah,
            "Daytime_evening_attendance": waktu_kuliah,
            "Displaced": merantau,
            "International": internasional,
            "Educational_special_needs": kebutuhan_khusus,
            "Mothers_qualification": pilihan["jenjang_ibu"][jenjang_ibu],
            "Fathers_qualification": pilihan["jenjang_ayah"][jenjang_ayah],
            "Mothers_occupation": pilihan["pekerjaan_ibu"][pekerjaan_ibu],
            "Fathers_occupation": pilihan["pekerjaan_ayah"][pekerjaan_ayah],
            "Tuition_fees_up_to_date": ukt_lancar,
            "Debtor": tunggakan,
            "Scholarship_holder": beasiswa,
        })
        if horizon == "tahun_pertama":
            masukan.update(akademik)

        baris = pd.DataFrame([masukan])
        nilai_risiko, sumbangan = faktor_pendorong(horizon, baris)
        pita_mahasiswa = str(pita(horizon, np.array([nilai_risiko])).iloc[0])

        st.markdown("---")
        kolom_hasil = st.columns([1.15, 1, 1])
        kartu_angka(kolom_hasil[0], "Peluang berhenti studi", f"{nilai_risiko * 100:.1f}%",
                    WARNA_PITA[pita_mahasiswa])
        kartu_angka(kolom_hasil[1], "Pita tindak lanjut", pita_mahasiswa,
                    WARNA_PITA[pita_mahasiswa])
        kartu_angka(kolom_hasil[2], "Ambang keputusan", f"{ambang(horizon):.2f}")
        batang_risiko(nilai_risiko, ambang(horizon))

        kolom_kiri, kolom_kanan = st.columns([1.25, 1])
        with kolom_kiri:
            st.markdown("**Faktor yang paling menggerakkan skor**")
            st.caption(
                "Dihitung dengan mengganti satu per satu kondisi mahasiswa ini dengan "
                "kondisi mahasiswa lazim, lalu mengukur selisih peluangnya."
            )
            if not sumbangan:
                st.info("Profil mahasiswa ini nyaris sama dengan mahasiswa lazim, "
                        "sehingga tidak ada faktor yang menonjol.")
            for kolom, selisih in sumbangan:
                arah = "naik" if selisih > 0 else "turun"
                kata = "menaikkan" if selisih > 0 else "menahan"
                st.markdown(
                    f"""
                    <div class="faktor {arah}">
                        <strong>{kk.nama(kolom)}</strong>: {jelaskan_nilai(kolom, masukan[kolom])}
                        <br><span style="color:{WARNA['kelabu']};font-size:0.84rem;">
                        {kata} peluang sebesar {abs(selisih) * 100:.1f} poin persen</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with kolom_kanan:
            st.markdown("**Tindak lanjut yang disarankan**")
            for saran in SARAN_PITA[pita_mahasiswa]:
                st.markdown(f"- {saran}")
            st.caption(
                "Skor ini adalah peringkat perhatian, bukan vonis. Keputusan akhir "
                "tetap berada pada dosen wali yang mengenal mahasiswanya."
            )


# ---------------------------------------------------------------------------
# Tab 2: antrean bimbingan
# ---------------------------------------------------------------------------
with tab_antrean:
    st.markdown("#### Menyusun antrean bimbingan sesuai kapasitas")
    st.caption(
        "Unggah berkas CSV berisi data mahasiswa, atau pakai berkas contoh berisi "
        "mahasiswa yang masih aktif. Kolomnya harus sama dengan dataset asli."
    )

    kolom_unggah, kolom_kapasitas = st.columns([1.4, 1])
    berkas = kolom_unggah.file_uploader("Berkas CSV mahasiswa", type=["csv"])
    pakai_contoh = kolom_unggah.checkbox("Pakai berkas contoh mahasiswa aktif", value=True)

    data_masuk = None
    if berkas is not None:
        isi = berkas.getvalue().decode("utf-8", "replace")
        pemisah = ";" if isi.count(";") > isi.count(",") else ","
        data_masuk = pd.read_csv(io.StringIO(isi), sep=pemisah)
    elif pakai_contoh:
        data_masuk = muat_contoh()

    if data_masuk is None:
        st.info("Belum ada data. Unggah berkas CSV atau centang pemakaian berkas contoh.")
    else:
        kolom_wajib = INFO["horizon"][horizon]["kolom_mentah"]
        hilang = [k for k in kolom_wajib if k not in data_masuk.columns]
        if hilang:
            st.error(
                f"Berkas kekurangan {len(hilang)} kolom yang dibutuhkan model "
                f"{'akhir tahun pertama' if horizon == 'tahun_pertama' else 'gerbang masuk'}: "
                + ", ".join(hilang[:8]) + ("..." if len(hilang) > 8 else "")
            )
        else:
            kapasitas = kolom_kapasitas.slider(
                "Kapasitas bimbingan bulan ini (jumlah mahasiswa)",
                min_value=5, max_value=min(300, len(data_masuk)),
                value=min(40, len(data_masuk)), step=5)
            kolom_kapasitas.caption(
                "Geser sesuai jumlah slot konselor yang benar benar tersedia. "
                "Antrean disusun dari puncak daftar, bukan dari seluruh mahasiswa terpanggil."
            )

            peluang = skor(horizon, data_masuk)
            hasil = data_masuk.copy()
            hasil.insert(0, "id_mahasiswa",
                         [f"MHS{n:04d}" for n in range(1, len(hasil) + 1)])
            hasil["program_studi"] = hasil.Course.map(kk.PROGRAM_STUDI)
            hasil["skor_risiko"] = peluang.round(4)
            hasil["pita"] = pita(horizon, peluang).to_numpy()
            hasil = hasil.sort_values("skor_risiko", ascending=False).reset_index(drop=True)
            hasil.insert(1, "peringkat", np.arange(1, len(hasil) + 1))

            di_atas_ambang = int((peluang >= ambang(horizon)).sum())
            kolom_ringkas = st.columns(4)
            kartu_angka(kolom_ringkas[0], "Mahasiswa dinilai", f"{len(hasil):,}")
            kartu_angka(kolom_ringkas[1], "Di atas ambang", f"{di_atas_ambang:,}",
                        WARNA["prioritas"])
            kartu_angka(kolom_ringkas[2], "Pita kritis",
                        f"{int((hasil.pita == 'Kritis').sum()):,}", WARNA["kritis"])
            kartu_angka(kolom_ringkas[3], "Terjadwal bulan ini", f"{kapasitas:,}",
                        WARNA["utama"])

            if di_atas_ambang > kapasitas:
                st.markdown(
                    f"""
                    <div class="kartu" style="border-left:4px solid {WARNA['prioritas']};">
                    Sebanyak {di_atas_ambang:,} mahasiswa berada di atas ambang, sementara
                    kapasitas bulan ini {kapasitas:,} orang. Antrean berikut sudah diurutkan
                    sehingga slot yang ada terpakai untuk mahasiswa dengan risiko tertinggi,
                    dan sisanya masuk daftar tunggu bulan berikutnya.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            kolom_tampil = ["peringkat", "id_mahasiswa", "program_studi", "skor_risiko", "pita",
                            "Age_at_enrollment", "Debtor", "Tuition_fees_up_to_date",
                            "Scholarship_holder"]
            if horizon == "tahun_pertama":
                kolom_tampil += ["Curricular_units_1st_sem_approved",
                                 "Curricular_units_2nd_sem_approved",
                                 "Curricular_units_2nd_sem_grade"]
            tampil = hasil.head(kapasitas)[kolom_tampil].rename(columns={
                "peringkat": "Peringkat", "id_mahasiswa": "ID", "program_studi": "Program studi",
                "skor_risiko": "Skor risiko", "pita": "Pita",
                "Age_at_enrollment": "Usia", "Debtor": "Tunggakan",
                "Tuition_fees_up_to_date": "UKT lancar", "Scholarship_holder": "Beasiswa",
                "Curricular_units_1st_sem_approved": "Lulus sem 1",
                "Curricular_units_2nd_sem_approved": "Lulus sem 2",
                "Curricular_units_2nd_sem_grade": "Nilai sem 2",
            })
            st.dataframe(tampil, width="stretch", hide_index=True,
                         column_config={"Skor risiko": st.column_config.ProgressColumn(
                             "Skor risiko", min_value=0.0, max_value=1.0, format="%.3f")})

            st.download_button(
                "Unduh seluruh hasil skoring sebagai CSV",
                data=hasil.to_csv(index=False, sep=";").encode("utf-8"),
                file_name="antrean_bimbingan.csv",
                mime="text/csv",
            )

            st.markdown("**Sebaran pita risiko pada berkas ini**")
            sebaran = (hasil.pita.value_counts()
                       .reindex(fa.PITA_RISIKO).fillna(0).astype(int))
            st.bar_chart(sebaran, color=WARNA["utama"], height=220)


# ---------------------------------------------------------------------------
# Tab 3: simulasi kebijakan
# ---------------------------------------------------------------------------
with tab_simulasi:
    st.markdown("#### Menakar dampak sebuah kebijakan")
    st.caption(
        "Simulasi mengubah satu kondisi pada seluruh mahasiswa di berkas contoh, "
        "lalu menghitung ulang skor risikonya dengan model yang sama."
    )

    contoh = muat_contoh()
    if contoh is None:
        st.info("Berkas contoh mahasiswa aktif tidak ditemukan di folder dataset.")
    else:
        kolom_pilih, kolom_hasil = st.columns([1, 1.4])
        with kolom_pilih:
            kebijakan = st.selectbox(
                "Kebijakan yang disimulasikan",
                [
                    "Menuntaskan seluruh tunggakan mahasiswa",
                    "Memastikan seluruh pembayaran uang kuliah lancar",
                    "Memberi beasiswa kepada mahasiswa berisiko tertinggi",
                    "Bimbingan akademik menaikkan kelulusan satu mata kuliah per semester",
                ],
            )
            cakupan = st.slider(
                "Cakupan kebijakan (persen mahasiswa berisiko tertinggi)",
                10, 100, 30, step=10,
                help="Kebijakan diterapkan hanya pada sekian persen teratas menurut skor "
                     "risiko saat ini.")
            jalankan = st.button("Jalankan simulasi", type="primary")

        if jalankan:
            sebelum = skor(horizon, contoh)
            urutan = np.argsort(-sebelum)
            jumlah_kena = max(1, int(len(contoh) * cakupan / 100))
            terpilih = contoh.index[urutan[:jumlah_kena]]

            sesudah_data = contoh.copy()
            if kebijakan.startswith("Menuntaskan"):
                sesudah_data.loc[terpilih, "Debtor"] = 0
            elif kebijakan.startswith("Memastikan"):
                sesudah_data.loc[terpilih, "Tuition_fees_up_to_date"] = 1
            elif kebijakan.startswith("Memberi beasiswa"):
                sesudah_data.loc[terpilih, "Scholarship_holder"] = 1
            else:
                for sebutan in ["1st", "2nd"]:
                    diambil = sesudah_data.loc[terpilih, f"Curricular_units_{sebutan}_sem_enrolled"]
                    lulus = sesudah_data.loc[terpilih, f"Curricular_units_{sebutan}_sem_approved"]
                    sesudah_data.loc[terpilih, f"Curricular_units_{sebutan}_sem_approved"] = (
                        np.minimum(lulus + 1, diambil))

            sesudah = skor(horizon, sesudah_data)
            naik_sebelum = int((sebelum >= ambang(horizon)).sum())
            naik_sesudah = int((sesudah >= ambang(horizon)).sum())

            with kolom_hasil:
                st.markdown("**Hasil simulasi**")
                kolom_angka = st.columns(3)
                kartu_angka(kolom_angka[0], "Rata rata risiko sebelum",
                            f"{sebelum.mean() * 100:.1f}%")
                kartu_angka(kolom_angka[1], "Rata rata risiko sesudah",
                            f"{sesudah.mean() * 100:.1f}%", WARNA["utama"])
                selisih = (sesudah.mean() - sebelum.mean()) * 100
                kartu_angka(kolom_angka[2], "Pergeseran", f"{selisih:+.1f} poin",
                            WARNA["aman"] if selisih < 0 else WARNA["kritis"])

                st.markdown(
                    f"Mahasiswa di atas ambang berubah dari **{naik_sebelum:,}** menjadi "
                    f"**{naik_sesudah:,}** orang, dari total {len(contoh):,} mahasiswa aktif. "
                    f"Kebijakan diterapkan pada {jumlah_kena:,} mahasiswa dengan risiko tertinggi."
                )
                perbandingan = pd.DataFrame({
                    "Sebelum": pd.Series(sebelum).groupby(
                        pita(horizon, sebelum)).size().reindex(fa.PITA_RISIKO).fillna(0),
                    "Sesudah": pd.Series(sesudah).groupby(
                        pita(horizon, sesudah)).size().reindex(fa.PITA_RISIKO).fillna(0),
                })
                st.bar_chart(perbandingan, height=240,
                             color=[WARNA["kelabu"], WARNA["utama"]])

            st.markdown(
                f"""
                <div class="catatan">
                Angka di atas adalah proyeksi berdasarkan pola yang dipelajari model dari
                data historis, bukan bukti sebab akibat. Model mengetahui bahwa mahasiswa
                tanpa tunggakan lebih jarang berhenti, tetapi ia tidak dapat memastikan
                bahwa menghapus tunggakan seseorang akan mengubah keputusannya. Sebagian
                sinyal keuangan justru merupakan gejala dari keputusan berhenti yang sudah
                diambil lebih dulu. Pakailah simulasi ini untuk menyusun urutan percobaan
                kebijakan, lalu ukur dampak sebenarnya lewat pembandingan kelompok.
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Tab 4: tentang model
# ---------------------------------------------------------------------------
with tab_model:
    st.markdown("#### Tentang kedua model")
    st.markdown(
        "Sistem ini memakai dua model yang menjawab pertanyaan sama pada dua titik waktu "
        "berbeda. Model gerbang masuk hanya boleh memakai data yang sudah ada saat "
        "mahasiswa diterima, sedangkan model akhir tahun pertama menambahkan catatan "
        "akademik dua semester pertama."
    )

    ringkasan = pd.DataFrame({
        ("Akhir tahun pertama" if h == "tahun_pertama" else "Gerbang masuk"): {
            "Algoritma": INFO["horizon"][h]["nama_model"],
            "Kalibrasi": INFO["horizon"][h]["kalibrasi"],
            "Jumlah kolom masukan": str(len(INFO["horizon"][h]["kolom_mentah"])
                                        + len(INFO["horizon"][h]["fitur_turunan"])),
            "Ambang keputusan": f"{INFO['horizon'][h]['ambang_keputusan']:.2f}",
            "Akurasi": f"{INFO['horizon'][h]['metrik_uji']['akurasi']:.3f}",
            "Presisi": f"{INFO['horizon'][h]['metrik_uji']['presisi']:.3f}",
            "Recall": f"{INFO['horizon'][h]['metrik_uji']['recall']:.3f}",
            "F1": f"{INFO['horizon'][h]['metrik_uji']['f1']:.3f}",
            "ROC-AUC": f"{INFO['horizon'][h]['metrik_uji']['roc_auc']:.3f}",
            "PR-AUC": f"{INFO['horizon'][h]['metrik_uji']['pr_auc']:.3f}",
            "Skor Brier": f"{INFO['horizon'][h]['metrik_uji']['brier']:.3f}",
        } for h in ["gerbang", "tahun_pertama"]
    })
    st.dataframe(ringkasan, width="stretch")
    st.caption(
        f"Seluruh angka dihitung pada {INFO['data']['baris_uji']:,} mahasiswa data uji "
        f"yang tidak pernah dipakai melatih maupun menyetel model. Data berlabel berjumlah "
        f"{INFO['data']['baris_berlabel']:,} mahasiswa, sedangkan "
        f"{INFO['data']['baris_aktif']:,} mahasiswa berstatus aktif sengaja dikeluarkan "
        "dari pelatihan karena hasil akhirnya belum diketahui."
    )

    kolom_a, kolom_b = st.columns(2)
    with kolom_a:
        st.markdown("**Sumbangan tiap blok informasi**")
        st.caption("Penurunan PR-AUC ketika seluruh kolom dalam satu blok diacak bersama.")
        blok = pd.DataFrame({
            ("Akhir tahun pertama" if h == "tahun_pertama" else "Gerbang masuk"):
                INFO["horizon"][h]["kepentingan_blok"]
            for h in ["gerbang", "tahun_pertama"]
        }).fillna(0)
        st.dataframe(blok.sort_values("Akhir tahun pertama", ascending=False),
                     width="stretch")

    with kolom_b:
        st.markdown("**Cara ambang keputusan ditentukan**")
        st.markdown(
            f"""
            Ambang tidak diambil dari angka bawaan 0,5, melainkan dari perhitungan biaya.
            Melewatkan satu calon dropout dinilai
            {INFO['biaya']['melewatkan_dropout']:.0f} kali lebih mahal daripada membimbing
            satu mahasiswa yang sebenarnya akan lulus. Ambang yang dipakai adalah titik
            dengan biaya harapan terendah menurut perbandingan tersebut, dihitung pada
            probabilitas luar lipatan data latih.

            Perbandingan biaya itu keputusan kebijakan pimpinan, bukan hasil perhitungan
            statistik. Menaikkannya membuat sistem lebih mudah curiga sehingga lebih banyak
            mahasiswa dipanggil, dan sebaliknya.
            """
        )

    st.markdown("**Batasan yang perlu diketahui pemakai**")
    st.markdown(
        """
        - Model memberi peringkat perhatian, bukan vonis. Mahasiswa berskor tinggi berarti
          pantas ditanyai kabarnya lebih dulu, bukan pasti akan berhenti.
        - Beberapa variabel berkaitan kuat dengan dropout tetapi tidak boleh dijadikan
          dasar perlakuan yang merugikan, khususnya jenis kelamin dan usia. Pemakaian yang
          benar adalah menambah dukungan, bukan mengurangi kesempatan.
        - Hubungan yang dipelajari model bersifat keterkaitan, bukan sebab akibat. Sebagian
          sinyal keuangan bisa jadi merupakan gejala dari keputusan berhenti yang sudah
          diambil, bukan penyebabnya.
        - Model perlu dilatih ulang setiap tahun akademik karena kurikulum, kebijakan biaya,
          dan profil pendaftar berubah dari waktu ke waktu.
        - Data latih berasal dari satu institusi pada rentang tahun tertentu, sehingga
          angkanya tidak otomatis berlaku untuk kampus lain.
        """
    )

    st.markdown(
        f"""
        <div class="catatan">
        Sumber data: <a href="{INFO['data']['sumber']}">{INFO['data']['sumber']}</a>.
        Model versi {INFO['identitas']['versi']}, dilatih ulang lewat notebook.ipynb pada
        proyek yang sama.
        </div>
        """,
        unsafe_allow_html=True,
    )
