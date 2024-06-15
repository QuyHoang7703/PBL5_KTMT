#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SoftwareSerial.h>

#define RX_PIN 11 // Chân RX của cổng nối tiếp mềm
#define TX_PIN 12 // Chân TX của cổng nối tiếp mềm

SoftwareSerial mySerial(RX_PIN, TX_PIN); // Tạo một đối tượng cổng nối tiếp mềm

LiquidCrystal_I2C lcd(0x27, 20, 4); // LCD 20x4 với địa chỉ I2C 0x27

int gateStatus = 0;
char incoming_value;

int parking_slots[] = {5, 6, 7, 8, 9, 10};

String status_sensor[] = {"_", "_", "_", "_", "_", "_"}; // trạng thái bãi đỗ xe : _ : empty , x : busy 
int num_status_sensor = sizeof(status_sensor) / sizeof(status_sensor[0]);
int total_slot; // tổng số vị trí 
int empty_slot; // số vị trí trống

unsigned long previousMillis = 0; // Biến lưu trữ thời gian trước đó
const long interval = 1000;        // Thời gian cập nhật là 1 giây (1000ms)

void setup() {
  Serial.begin(9600);
  mySerial.begin(9600); // Khởi động cổng nối tiếp mềm với baud rate 9600
  
  total_slot = sizeof(parking_slots) / sizeof(parking_slots[0]);
  empty_slot = total_slot; // khi khởi động thì mọi vị trí đều trống, empty = total_slot

  for (int i = 0; i < total_slot; i++) {
    pinMode(parking_slots[i], INPUT_PULLUP);
  }

  lcd.init();
  lcd.backlight();
}

void loop() {
  unsigned long currentMillis = millis(); // Lấy thời gian hiện tại

  // Kiểm tra nếu đã đủ thời gian để cập nhật LCD
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis; // Lưu lại thời gian gần nhất

    checkParkingSlots();          // Cập nhật trạng thái bãi đỗ xe
    displayLCD();                 // Hiển thị trạng thái lên LCD
    sendParkingStatusToESP8266(); // Gửi trạng thái bãi đỗ xe đến ESP8266
  }

  // Xử lý các công việc khác ở đây (nếu có)
}

void checkParkingSlots() {
  empty_slot = total_slot; // Khởi tạo số lượng chỗ trống là tổng số chỗ

  for (int i = 0; i < total_slot; i++) {
    if (digitalRead(parking_slots[i]) == LOW) {
      status_sensor[i] = "x";
      empty_slot -= 1;
    } else {
      status_sensor[i] = "_";
    }
  }
}

void sendParkingStatusToESP8266() {
  Serial.print("empty_slot: ");
  Serial.println(empty_slot); // In ra giá trị empty_slot trước khi gửi đi
  String message = "PARKING_STATUS";
  for (int i = 0; i < total_slot; i++) {
    message += "|" + status_sensor[i];
  }
  message += "|" + String(empty_slot); // Thêm số lượng chỗ trống vào thông điệp
  mySerial.println(message); // Gửi thông điệp đến ESP8266
  Serial.println("Sent to ESP8266: " + message); // Debug: in thông điệp gửi đi
}

void displayLCD() {
  lcd.clear();
  lcd.setCursor(2, 0);
  lcd.print("Con trong: ");
  lcd.print(empty_slot);
  lcd.setCursor(0, 1);
  lcd.print("P1: ");
  lcd.print(status_sensor[0]);
  lcd.print(" ");
  lcd.print(status_sensor[1]);
  lcd.print(" ");
  lcd.print(status_sensor[2]);
  lcd.print(" ");
  lcd.setCursor(0, 2);
  lcd.print("P2: ");
  lcd.print(status_sensor[3]);
  lcd.print(" ");
  lcd.print(status_sensor[4]);
  lcd.print(" ");
  lcd.print(status_sensor[5]);
  lcd.print(" ");
}
