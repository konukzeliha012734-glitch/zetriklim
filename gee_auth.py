"""Zetriklim için bir defalık Google Earth Engine kullanıcı doğrulaması."""

import argparse

from zetriklim.gee import authenticate_localhost


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Google Cloud / Earth Engine Project ID")
    args = parser.parse_args()
    print(f"Earth Engine projesi: {args.project}")
    print("Tarayıcı açıldığında Google hesabınızla izin verin.")
    authenticate_localhost(args.project)
    print("Earth Engine bağlantısı başarıyla kaydedildi.")
