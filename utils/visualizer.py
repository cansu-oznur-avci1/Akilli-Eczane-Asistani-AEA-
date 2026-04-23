import os
import base64
import requests

def export_graph_mermaid_png(mermaid_str: str, output_path: str = "AEA_flow.png") -> None:
    """
    Mermaid grafiği kodunu alır ve mermaid.ink API'sini kullanarak PNG olarak kaydeder.
    Mermaid-CLI gerektirmez.
    """
    try:
        # mermaid.ink doğrudan base64 bekliyor
        encoded_str = base64.b64encode(mermaid_str.encode("utf-8")).decode("utf-8")
        url = f"https://mermaid.ink/img/{encoded_str}"
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Mermaid PNG grafiği {output_path} dosyasına başarıyla kaydedildi.")
        else:
            print(f"Mermaid PNG dönüştürme başarısız oldu: HTTP {response.status_code}")
    except Exception as e:
        print(f"Mermaid PNG dönüştürme hatası (İnternet veya API bağlantı sorunu): {e}")

if __name__ == "__main__":
    test_str = "graph TD;\n    A-->B;\n    A-->C;\n    B-->D;\n    C-->D;"
    export_graph_mermaid_png(test_str, "test.png")
