import os
import zipfile
import gdown

def download_and_unzip_chroma(url, extract_to="chroma_ve"):
    """
    Descargar el ZIP file usando la url y extraerlo en la ruta especificada..
    """
    if os.path.exists(extract_to):
        print("✅ Vectorstore ya esta presente.")
        return

    zip_path = "chroma_ve.zip"
    print("🔽 Descargar Chroma vectorstore...")
    gdown.download(drive_url, zip_path, quiet=False)

    print("📦 Descomprimir...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)
    print("✅ Vectorstore extraido en:", extract_to)