from ultralytics import YOLO
import traceback
import torch


# Lista de modelos e batchs iniciais (com base nas mainhas especificações do computador)
model_settings = {
    "yolo11n.yaml": 128,
    "yolo11s.yaml": 80,
    "yolo11m.yaml": 48,
    "yolo8n.yaml": 128,   
    "yolo8s.yaml": 80,    
    "yolo8m.yaml": 48,    
    "yolo5n.yaml": 128,   
    "yolo5s.yaml": 80,    
    "yolo5m.yaml": 48     
}

# Configurações gerais
data_config = "config.yaml"
epochs = 100
imgsz = 416
project_name = "resultados_treinamento"
min_batch = 8  # Limiar mínimo de segurança para batch size

for model_path, initial_batch in model_settings.items():
    model_name = model_path.replace(".yaml", "")
    batch = initial_batch
    success = False

    print(f"\n🔧 Iniciando treinamento para {model_name} com batch {batch}")

    while not success and batch >= min_batch:
        try:
            model = YOLO(model_path)
            results = model.train(
                data=data_config,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                project=project_name,
                name=model_name
            )
            print(f"✅ Treinamento concluído para {model_name} com batch {batch}")
            success = True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"⚠️ Erro de memória com batch {batch} em {model_name}. Reduzindo pela metade...")
                batch = batch // 2
                torch.cuda.empty_cache()  # limpa memória da GPU entre tentativas
            else:
                print(f"❌ Erro inesperado ao treinar {model_name}: {e}")
                traceback.print_exc()
                break
        except Exception as e:
            print(f"❌ Erro geral ao treinar {model_name}: {e}")
            traceback.print_exc()
            break

    if not success:
        print(f"🚫 Treinamento de {model_name} falhou mesmo após reduzir o batch.")
