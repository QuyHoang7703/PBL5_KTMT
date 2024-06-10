// #include <ESP8266WiFi.h>
// #include <ESP8266WebServer.h>
// #include <Servo.h> 

// ESP8266WebServer server(80);

// Servo servoVao;
// Servo servoRa;

// const int IR_PIN_VAO = D2;
// const int IR_PIN_RA = D3; 

// const char* ssid = "metmoiqua";
// const char* password = "11111111";

// void setup() {
//   Serial.begin(9600);

//   servoVao.attach(D0, 500, 2400);
//   servoRa.attach(D1, 500, 2400);
//   servoVao.write(0);
//   servoRa.write(0);

//   pinMode(IR_PIN_VAO, INPUT);
//   pinMode(IR_PIN_RA, INPUT);

//   WiFi.begin(ssid, password);
//   while (WiFi.status() != WL_CONNECTED) {
//     delay(1000);
//     Serial.println("Connecting to WiFi..");
//   }
//   Serial.println("Connected to WiFi");
//   Serial.println("IP address: ");
//   Serial.println(WiFi.localIP());

//   server.on("/open_door_vao", handleOpenDoorvao);
//   server.on("/open_door_ra", handleOpenDoorra);
//   server.begin();
//   Serial.println("HTTP server started");
// }

// void handleOpenDoorvao() {
//   Serial.println("Mở cửa vào");
//   servoVao.write(90);
//   server.send(200, "text/plain", "Door opening...");  
// }

// void handleOpenDoorra() {
//   Serial.println("Mở cửa ra");
//   servoRa.write(90);
//   server.send(200, "text/plain", "Door opening...");  
// }

// void loop() {
//   server.handleClient();

//   // Kiểm tra trạng thái của cảm biến hồng ngoại
//   bool currentCarPresentVao = digitalRead(IR_PIN_VAO) == HIGH;
//   bool currentCarPresentRa = digitalRead(IR_PIN_RA) == HIGH;

//   if (!currentCarPresentVao && servoVao.read() == 90) {
//     Serial.println("Đóng cửa vào");
//     servoVao.write(0);
//   }

//   if (!currentCarPresentRa && servoRa.read() == 90) {
//     Serial.println("Đóng cửa ra");
//     servoRa.write(0);
//   }
// }
#include <SoftwareSerial.h>
#include <ESP8266WiFi.h> // hoặc #include <WiFi.h> nếu bạn sử dụng ESP32
#include <ESP8266WebServer.h>
#include <Servo.h> 

ESP8266WebServer server(80);

Servo servoVao;
Servo servoRa;

// Thông tin mạng Wi-Fi
const char* ssid = "metmoiqua";
const char* password = "11111111";

void setup() {
  Serial.begin(9600);
  
  servoVao.attach(D0,500,2400); // Sửa tên đối tượng servo
  servoRa.attach(D1,500,2400); // Sửa tên đối tượng servo
  servoVao.write(0);
  servoRa.write(0);

  // Kết nối đến mạng Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi..");
  }
  Serial.println("Connected to WiFi");



  Serial.println("IP address: ");
  Serial.println(WiFi.localIP());

  // Thiết lập server để lắng nghe yêu cầu mở cửa
  server.on("/open_door_vao", handleOpenDoorvao);
  server.on("/open_door_ra", handleOpenDoorra);
  server.begin();
  Serial.println("HTTP server started");


}

void handleOpenDoorvao() {
  Serial.println("Mở cửa vào");
  servoVao.write(90);  // Quay servo để mở cửa
  delay(2000);        // Giữ cửa mở trong 5 giây
  servoVao.write(0);
  server.send(200, "text/plain", "Door opened");  // Gửi phản hồi về cho client

}

void handleOpenDoorra() {
  Serial.println("Mở cửa ra");
  servoRa.write(90);  // Quay servo để mở cửa
  delay(2000);        // Giữ cửa mở trong 5 giây
  servoRa.write(0);
  server.send(200, "text/plain", "Door opened");  // Gửi phản hồi về cho client
}

void loop() {
  server.handleClient();
}