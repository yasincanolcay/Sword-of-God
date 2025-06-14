import cv2
from ultralytics import YOLO
import time

#İyi seviye bir sistemde hızlı çalşacaktır, güçlü bir ekran kartı tavsiye edilir
#Ekran kartı yoksa işlemciyi kullanacaktır, uzun süre çalışırsa bilgisayarı yorabilir
# YOLO modelini yükle ve CUDA varsa kullan
model = YOLO("yolov8n.pt") #bir çok cihazda çalışabilir basit modeldir
model.to('cuda' if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu')#ekran kartı varsa onu kullan

cap = cv2.VideoCapture(0)  # 0 default kamera, bilgisayara bağlı kameraları seçin 0-1-2 vs...
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

auto_mode = False
multiple_mode = False
locked_targets = []  # Çoklu hedef listesi
prev_sizes = {}
cursor_x, cursor_y = 320, 240
dangerous_labels = ["knife", "gun", "person"]
system_status = "PASIF"
ammo = 20 #mermi sayısı
results = None
frame_count = 0

def is_point_in_box(px, py, box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return x1 <= px <= x2 and y1 <= py <= y2

def risk_level(growth):
    if growth < 0.05:
        return "DÜŞÜK", (255, 255, 255), 0
    elif growth < 0.25:
        return "ORTA - UYAR", (0, 255, 255), 1
    else:
        return "YÜKSEK - YOK ET", (0, 0, 255), 2

def is_hands_up(box, frame):
    return False  # Teslim olma algılama şimdilik yok

def mouse_callback(event, x, y, flags, param):
    global locked_targets, results, multiple_mode
    if event == cv2.EVENT_LBUTTONDOWN and results is not None:
        for idx, box in enumerate(results.boxes):
            if is_point_in_box(x, y, box):
                if multiple_mode:
                    if idx not in locked_targets:
                        locked_targets.append(idx)
                        print(f"Multiple Mode: Hedef eklendi, ID {idx}")
                else:
                    locked_targets = [idx]  # Tek hedef seç
                    print(f"Tek hedef mouse ile seçildi: ID {idx}")
                break

cv2.namedWindow("Akilli Sistem")
cv2.setMouseCallback("Akilli Sistem", mouse_callback)

print("""
[Komutlar]
T = En yakın hedefe kilitlen (Multiple Mode kapalıyken)
X = Tüm hedef takibini bırak (Multiple Mode ve Auto Mode'dan)
V = Kilitli hedefleri sırayla vur (Multiple Mode'da tüm hedefleri vurur)
R = Mermiyi doldur (20 olur)
A = Auto Mode Aç/Kapat (Multiple Mode kapatılır)
M = Multiple Mode Aç/Kapat (Auto Mode kapatılır)
Q = Çık
Fare ile Multiple Mode'da hedef seçebilirsin.
""")

def fire_weapon(target_id):
    global ammo, system_status
    if ammo > 0:
        ammo -= 1
        system_status = f"MANUEL VUR - Hedef: {target_id} - Mermi: {ammo}"
        print(f"Hedef {target_id} vuruldu! Mermi kaldı: {ammo}")
    else:
        system_status = "MERMI BİTTİ! R basarak doldur"
        print(system_status)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    frame_count += 1
    #if içindeki 1 her 1 saniyede analiz eder,pc kasıyorsa bu sayıyı arttırın(sayı artarsa analiz yanlış yapılabilir)
    #(2-3) saniye sonrası analiz yavaşlar ve doğru çalışmaz
    if frame_count % 1 == 0:
        results = model(frame, verbose=False)[0]

    highest_risk_level = 0
    risk_text = "BILINMIYOR"
    risk_color = (255, 255, 255)

    if results:
        for idx, box in enumerate(results.boxes):
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = box.conf[0].item()
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            current_size = (x2 - x1) * (y2 - y1)
            previous_size = prev_sizes.get(idx, current_size)
            growth = (current_size - previous_size) / max(previous_size, 1)
            prev_sizes[idx] = current_size

            level, level_color, risk_val = risk_level(growth)
            if risk_val > highest_risk_level:
                highest_risk_level = risk_val

            # Renk ayarı: hedef kilitli ise sarı, değilse risk seviyesine göre
            if idx in locked_targets:
                color = (255, 255, 0)  # Sarı
                risk_text = level
                risk_color = level_color
                dx = cx - cursor_x
                dy = cy - cursor_y
                cursor_x += int(dx * 0.05)
                cursor_y += int(dy * 0.05)
            else:
                # Risk seviyesine göre renk
                if risk_val == 2:
                    color = (0, 0, 255)  # Kırmızı yüksek risk
                elif risk_val == 1:
                    color = (0, 255, 255)  # Orta risk
                else:
                    color = (0, 255, 0)  # Düşük risk

            # Auto mode vurma işlemi sadece Auto Mode aktifken ve multiple_mode kapalıyken
            if auto_mode and not multiple_mode and ammo > 0:
                if label in dangerous_labels and growth > 0.25 and not is_hands_up(box, frame):
                    ammo -= 1
                    system_status = f"AUTO MODE: {level} - Mermi: {ammo}"
                    color = (0, 0, 255)  # Kırmızı vurma sırasında
                elif ammo == 0:
                    system_status = "MERMI BİTTİ! R basarak doldur"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.drawMarker(frame, (cursor_x, cursor_y), (0, 0, 255),
                   markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
    #sağdaki siyah nalanda bilgi paneli
    hud_width = 300
    padded_frame = cv2.copyMakeBorder(frame, 0, 0, 0, hud_width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    text_x = width + 10

    cv2.putText(padded_frame, f"Auto Mode: {'ON' if auto_mode else 'OFF'}", (text_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(padded_frame, f"Multiple Mode: {'ON' if multiple_mode else 'OFF'}", (text_x, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    locked_text = ','.join(map(str, locked_targets)) if locked_targets else "Yok"
    cv2.putText(padded_frame, f"Kilitli Hedefler: {locked_text}", (text_x, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(padded_frame, f"Risk: {risk_text}", (text_x, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 1)
    cv2.putText(padded_frame, f"Durum: {system_status}", (text_x, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 1)
    cv2.putText(padded_frame, f"Mermi: {ammo}/20", (text_x, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Akilli Sistem", padded_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("t"):
        if not multiple_mode and results and len(results.boxes) > 0:
            locked_targets = [0]
            print("Tek hedef kilitlendi: ID 0")
    elif key == ord("x"):
        locked_targets = []
        system_status = "Tüm hedef takibi bırakıldı"
        print(system_status)
    elif key == ord("a"):
        auto_mode = not auto_mode
        if auto_mode and multiple_mode:
            multiple_mode = False
        system_status = f"Auto Mode: {'AÇIK' if auto_mode else 'KAPALI'}"
        print(system_status)
    elif key == ord("m"):
        multiple_mode = not multiple_mode
        if multiple_mode and auto_mode:
            auto_mode = False
        locked_targets = []  # Yeni moda geçince kilitli hedefleri sıfırla
        system_status = f"Multiple Mode: {'AÇIK' if multiple_mode else 'KAPALI'}"
        print(system_status)
    elif key == ord("r"):
        ammo = 20
        system_status = "Mermi dolduruldu: 20"
        print(system_status)
    elif key == ord("v"):
        if locked_targets:
            current_target = locked_targets.pop(0)
            fire_weapon(current_target)
        else:
            system_status = "Vurulacak hedef yok!"
            print(system_status)

cap.release()
cv2.destroyAllWindows()
