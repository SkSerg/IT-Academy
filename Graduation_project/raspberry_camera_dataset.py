import cv2

# Инициализация камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадров из видеопотока
    ret, frame = cap.read()

    # Если кадр считан корректно, ret будет True
    if not ret:
        print("Не удалось получить кадр. Попробуйте еще раз")
        break

    # Отображение кадра
    cv2.imshow('Camera Output', frame)

    # Закрытие окна при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов и закрытие окон
cap.release()
cv2.destroyAllWindows()