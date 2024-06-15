#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <SoftwareSerial.h>
#include <Servo.h>

const char* ssid = "Trung Nhan";      // Tên mạng WiFi của bạn
const char* password = "0929359373";  // Mật khẩu WiFi của bạn

const int RX_PIN = D6;  // Chân RX của ESP8266 (kết nối với TX của Arduino)
const int TX_PIN = D7;  // Chân TX của ESP8266 (kết nối với RX của Arduino)

SoftwareSerial mySerial(RX_PIN, TX_PIN);  // Tạo đối tượng SoftwareSerial

ESP8266WebServer server(80);  // Tạo đối tượng ESP8266WebServer lắng nghe cổng 80

Servo servoVao;
Servo servoRa;

// Các biến cho dữ liệu từ Arduino
int empty_slot = 0;
String status_sensor[] = { "_", "_", "_", "_", "_", "_" };  // Mảng trạng thái bãi đỗ xe

void setup() {
  Serial.begin(9600);    // Khởi động Serial Monitor
  mySerial.begin(9600);  // Khởi động cổng nối tiếp mềm với baud rate 9600

  // Kết nối WiFi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Đang kết nối đến WiFi...");
  }
  Serial.println("WiFi đã kết nối");
  Serial.println("Địa chỉ IP: ");
  Serial.println(WiFi.localIP());

  // Thiết lập các route của server
  server.on("/", handleRoot);                      // Route chính để hiển thị trạng thái bãi đỗ xe
  server.on("/open_door_vao", handleOpenDoorvao);  // Route để mở cửa vào
  server.on("/open_door_ra", handleOpenDoorra);    // Route để mở cửa ra
  server.begin();                                  // Bắt đầu máy chủ web
  Serial.println("Máy chủ HTTP đã khởi động");

  // Kết nối và thiết lập servo
  servoVao.attach(D0, 500, 2400);  // Chân D0 cho servo vào
  servoRa.attach(D1, 500, 2400);   // Chân D1 cho servo ra
  servoVao.write(0);               // Đóng cửa vào khi khởi động
  servoRa.write(0);                // Đóng cửa ra khi khởi động
}

void loop() {
  server.handleClient();  // Xử lý các yêu cầu từ client

  // Đọc dữ liệu từ Arduino qua SoftwareSerial
  if (mySerial.available() > 0) {
    String message = mySerial.readStringUntil('\n');  // Đọc dữ liệu từ Serial
    Serial.println("Nhận từ Arduino: " + message);    // Debug: in dữ liệu nhận được
    parseMessage(message);                            // Phân tích và cập nhật dữ liệu từ Arduino
  }
}

// Hàm xử lý route chính
void handleRoot() {
  String html = "<html><body>";
  html += "<h1>Trang thai bai do xe</h1>";
  html += "<p>So cho trong: <span id='empty_slot'>" + String(empty_slot) + "</span></p>";
  html += "<p>Trang thai cac vi tri:</p>";
  html += "<ul>";
  for (int i = 0; i < 6; i++) {
    html += "<li>Vi tri " + String(i + 1) + ": <span id='slot_" + String(i + 1) + "'>" + (status_sensor[i] == "x" ? "Dang su dung" : "Con trong") + "</span></li>";
  }
  html += "</ul>";
  html += "<p><a href='/open_door_vao'>Mo cua vao</a></p>";
  html += "<p><a href='/open_door_ra'>Mo cua ra</a></p>";
  html += "<script>";
  html += "setInterval(function() {";
  html += "  fetch('/status').then(response => response.text()).then(data => {";
  html += "    document.getElementById('status').innerHTML = data;";  // Cập nhật toàn bộ HTML khi nhận được dữ liệu mới
  html += "  });";
  html += "}, 2000);";  // Cập nhật mỗi 2 giây
  html += "</script>";
  html += "</body></html>";

  server.send(200, "text/html", html);  // Gửi phản hồi HTML về client
}

// Hàm phân tích và cập nhật dữ liệu từ Arduino
void parseMessage(String message) {
  // Dữ liệu từ Arduino có định dạng "PARKING_STATUS_SEND|_|_|_|x|_|_|5"
  if (message.startsWith("PARKING_STATUS")) {
    int pos1 = message.indexOf('|');
    int pos2 = message.lastIndexOf('|');

    if (pos1 != -1 && pos2 != -1 && pos2 > pos1) {
      String data = message.substring(pos1 + 1, pos2);
      empty_slot = message.substring(pos2 + 1).toInt();  // Cập nhật empty_slot từ dữ liệu nhận được

      Serial.println("Chuỗi data nhận được từ Arduino:");
      Serial.println(data);  // In chuỗi data lên Serial Monitor

      Serial.println("Cập nhật trạng thái các vị trí:");
      for (int i = 0; i < 6; i++) {
        char status_char = data.charAt(i * 2);  // Lấy ký tự tại vị trí i * 2 + 1 trong chuỗi data
        // status_sensor[i] = status_char == 'x' ? "Busy" : "Empty"; // Cập nhật mảng status_sensor
        status_sensor[i] = status_char;
        Serial.print("Vị trí " + String(i + 1) + ": " + status_sensor[i] + " ");  // Debug: in từng trạng thái của vị trí đỗ xe
        Serial.print(" ");                                                        // In một khoảng trống để phân tách
      }
      Serial.println();

      // Sau khi cập nhật, gửi lại toàn bộ HTML
      updateClient();
    }
  }
}

// Hàm gửi toàn bộ HTML về client
void updateClient() {
  String html = "<html><body>";
  html += "<h1>Trang thai bai do</h1>";
  html += "<p>So cho trong: <span id='empty_slot'>" + String(empty_slot) + "</span></p>";
  html += "<p>Trang thai cac vi tris:</p>";
  html += "<ul>";
  for (int i = 0; i < 6; i++) {
    html += "<li>Vi tri " + String(i + 1) + ": <span id='slot_" + String(i + 1) + "'>" + (status_sensor[i] == "x" ? "Dang su dung" : "Con trong") + "</span></li>";
  }
  html += "</ul>";
  html += "<p><a href='/open_door_vao'>Mo cua vao</a></p>";
  html += "<p><a href='/open_door_ra'>Mo cua ra</a></p>";
  html += "<script>";
  html += "setInterval(function() {";
  html += "  fetch('/status').then(response => response.text()).then(data => {";
  html += "    document.getElementById('status').innerHTML = data;";  // Cập nhật toàn bộ HTML khi nhận được dữ liệu mới
  html += "  });";
  html += "}, 2000);";  // Cập nhật mỗi 2 giây
  html += "</script>";
  html += "</body></html>";

  server.send(200, "text/html", html);  // Gửi phản hồ
}

// Hàm xử lý yêu cầu mở cửa vào
void handleOpenDoorvao() {
  Serial.println("Mở cửa vào");
  servoVao.write(90);  // Quay servo để mở cửa
  delay(5000);         // Giữ cửa mở trong 2 giây
  servoVao.write(0);
  updateClient();                                    // Cập nhật trạng thái bãi đỗ xe sau khi mở cửa
  server.send(200, "text/plain", "Cửa đã được mở");  // Gửi phản hồi về cho client
}

// Hàm xử lý yêu cầu mở cửa ra
void handleOpenDoorra() {
  Serial.println("Mở cửa ra");
  servoRa.write(90);  // Quay servo để mở cửa
  delay(5000);        // Giữ cửa mở trong 2 giây
  servoRa.write(0);
  updateClient();                                    // Cập nhật trạng thái bãi đỗ xe sau khi mở cửa
  server.send(200, "text/plain", "Cửa đã được mở");  // Gửi phản hồi về cho client
}