import cv2
import numpy as np

# Initialize video capture from video camera
cap = cv2.VideoCapture(0) 

# Load ONNX model
model = cv2.dnn.readNetFromONNX("MobileNet_2_1.onnx")

# Set video resolution 
cap.set(3, 640)  
cap.set(4, 480)

while True:
    
    # Read frame from video 
    _, frame = cap.read()  
    
    # Resize frame to model input size
    resized = cv2.resize(frame, (224, 224))
    
    # Change color channel order from BGR to RGB 
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # Convert to numpy array
    img_np = np.array(resized)
    
    # Normalize pixel values 
    img_np = img_np.transpose((2, 0, 1)) / 255.0  
    img_np = img_np.reshape(1, *img_np.shape)          # Add batch dimension
            
    
    # Make prediction 
    model.setInput(img_np)     
    output = model.forward()
            
    # Get predicted class with highest confidence 
    class_id = np.argmax(output)
            
    # Display text on frame with predicted class 
    cv2.putText(frame, str(class_id), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            
    # Show the resulting frame    
    cv2.imshow('Frame', frame) 
        
    # Press 'q' to exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything 
cap.release()
cv2.destroyAllWindows()  