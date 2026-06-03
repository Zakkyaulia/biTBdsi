CREATE TABLE IF NOT EXISTS dim_mahasiswa (
    mahasiswa_key integer PRIMARY KEY,
    nim varchar(32),
    nama varchar(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_waktu (
    waktu_key integer PRIMARY KEY,
    tanggal_lulus date,
    hari integer,
    bulan integer,
    tahun integer
);

CREATE TABLE IF NOT EXISTS dim_periode_wisuda (
    periode_key integer PRIMARY KEY,
    tahun_wisuda integer NOT NULL,
    periode_num integer NOT NULL,
    periode_label varchar(32) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_ipk (
    ipk_key integer PRIMARY KEY,
    ipk_numeric numeric(4, 2) NOT NULL,
    rentang_ipk varchar(32) NOT NULL,
    predikat_ipk varchar(64) NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_lama_studi (
    lama_studi_key integer PRIMARY KEY,
    lama_studi_bulan integer NOT NULL,
    kategori_lama_studi varchar(64) NOT NULL,
    flag_tepat_waktu boolean NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_tugas_akhir (
    ta_key integer PRIMARY KEY,
    judul_tugas_akhir text NOT NULL,
    judul_preprocessed text,
    judul_final text,
    skor_kemiripan_tertinggi numeric(8, 6),
    kategori_keunikan varchar(64),
    ta_key_termirip integer,
    pca_x numeric(14, 8),
    pca_y numeric(14, 8),
    CONSTRAINT fk_dim_ta_termirip
        FOREIGN KEY (ta_key_termirip)
        REFERENCES dim_tugas_akhir (ta_key)
        DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS dim_dosen (
    dosen_key integer PRIMARY KEY,
    nama_dosen_normalized varchar(255) NOT NULL UNIQUE,
    nama_asal varchar(255)
);

CREATE TABLE IF NOT EXISTS fact_kelulusan (
    kelulusan_key integer PRIMARY KEY,
    mahasiswa_key integer NOT NULL REFERENCES dim_mahasiswa (mahasiswa_key),
    waktu_key integer NOT NULL REFERENCES dim_waktu (waktu_key),
    periode_key integer NOT NULL REFERENCES dim_periode_wisuda (periode_key),
    ipk_key integer NOT NULL REFERENCES dim_ipk (ipk_key),
    lama_studi_key integer NOT NULL REFERENCES dim_lama_studi (lama_studi_key),
    ta_key integer NOT NULL REFERENCES dim_tugas_akhir (ta_key),
    ipk numeric(4, 2) NOT NULL,
    lama_studi_bulan integer NOT NULL,
    flag_tepat_waktu boolean NOT NULL,
    jumlah_record integer NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS bridge_peran_dosen (
    peran_dosen_key integer PRIMARY KEY,
    kelulusan_key integer NOT NULL REFERENCES fact_kelulusan (kelulusan_key),
    dosen_key integer NOT NULL REFERENCES dim_dosen (dosen_key),
    jenis_peran varchar(32) NOT NULL,
    urutan_peran integer NOT NULL,
    role_count integer NOT NULL DEFAULT 1
);

CREATE OR REPLACE VIEW vw_dashboard_kelulusan AS
SELECT
    f.kelulusan_key,
    m.mahasiswa_key,
    m.nim,
    m.nama,
    w.waktu_key,
    w.tanggal_lulus,
    w.hari,
    w.bulan,
    w.tahun AS tahun_lulus,
    p.periode_key,
    p.tahun_wisuda,
    p.periode_num,
    p.periode_label,
    i.ipk_key,
    i.rentang_ipk,
    i.predikat_ipk,
    l.lama_studi_key,
    l.kategori_lama_studi,
    l.flag_tepat_waktu,
    ta.ta_key,
    ta.judul_tugas_akhir,
    ta.judul_preprocessed,
    ta.judul_final,
    ta.skor_kemiripan_tertinggi,
    ta.kategori_keunikan,
    ta.ta_key_termirip,
    ta_ref.judul_tugas_akhir AS judul_termirip,
    m_ref.nama AS pemilik_judul_termirip,
    ta.pca_x,
    ta.pca_y,
    f.ipk,
    f.lama_studi_bulan,
    f.jumlah_record
FROM fact_kelulusan f
JOIN dim_mahasiswa m ON m.mahasiswa_key = f.mahasiswa_key
JOIN dim_waktu w ON w.waktu_key = f.waktu_key
JOIN dim_periode_wisuda p ON p.periode_key = f.periode_key
JOIN dim_ipk i ON i.ipk_key = f.ipk_key
JOIN dim_lama_studi l ON l.lama_studi_key = f.lama_studi_key
JOIN dim_tugas_akhir ta ON ta.ta_key = f.ta_key
LEFT JOIN dim_tugas_akhir ta_ref ON ta_ref.ta_key = ta.ta_key_termirip
LEFT JOIN fact_kelulusan f_ref ON f_ref.ta_key = ta_ref.ta_key
LEFT JOIN dim_mahasiswa m_ref ON m_ref.mahasiswa_key = f_ref.mahasiswa_key;

CREATE OR REPLACE VIEW vw_dashboard_peran_dosen AS
SELECT
    b.peran_dosen_key,
    b.kelulusan_key,
    b.dosen_key,
    d.nama_dosen_normalized,
    d.nama_asal,
    b.jenis_peran,
    b.urutan_peran,
    b.role_count,
    p.tahun_wisuda,
    p.periode_num,
    p.periode_label,
    f.ipk,
    f.lama_studi_bulan
FROM bridge_peran_dosen b
JOIN dim_dosen d ON d.dosen_key = b.dosen_key
JOIN fact_kelulusan f ON f.kelulusan_key = b.kelulusan_key
JOIN dim_periode_wisuda p ON p.periode_key = f.periode_key;
