// -----------------------------
// ESP32 I/O Bridge para ped_crosswalk_tag.py
// Protocolo (115200 8N1):
//   PC -> ESP: OUT,<b0..b4>   (veh_g,veh_y,veh_r,ped_w,ped_dw)  '1' = LIGADO
//   PC -> ESP: PING           (ESP responde PONG)
//   ESP -> PC: TAG,<uidhex>   (leitura RC522)
//   ESP -> PC: BTN,PED        (botão analógico pressionado)
//   ESP -> PC: HB             (heartbeat 1s)
// LEDs são ATIVOS EM LOW (LOW = ligado).
// -----------------------------

#include <SPI.h>
#include <MFRC522.h>

// ---------- Pinos LEDs ----------
const int ledVermelhoCarro    = 18;
const int ledAmareloCarro     = 19;
const int ledVerdeCarro       = 21;
const int ledVermelhoPedestre = 22;
const int ledVerdePedestre    = 23;

// ---------- Botão analógico ----------
const int pinoBotaoAnalogico  = 34;   // ADC somente entrada
const int THRESH_BOTAO        = 100;  // ajuste conforme hardware
const uint32_t DEBOUNCE_BTN_MS = 800; // antirruído

// ---------- SPI do RC522 (remapeado) ----------
const int RFID_SS   = 25;  // SDA/SS
const int RFID_RST  = 4;   // RST
const int RFID_SCK  = 14;  // SCK
const int RFID_MISO = 27;  // MISO
const int RFID_MOSI = 26;  // MOSI

MFRC522 mfrc522(RFID_SS, RFID_RST);

// ---------- Lógica ----------
#define ON   LOW
#define OFF  HIGH

// ---------- Timers ----------
const uint32_t HEARTBEAT_MS = 1000;

// ---------- Estado ----------
String rxLine;
uint32_t lastHB = 0;
uint32_t lastBtnMs = 0;

// Utilidades LEDs
void setCar(bool redOn, bool yellowOn, bool greenOn) {
  digitalWrite(ledVermelhoCarro,  redOn    ? ON : OFF);
  digitalWrite(ledAmareloCarro,   yellowOn ? ON : OFF);
  digitalWrite(ledVerdeCarro,     greenOn  ? ON : OFF);
}

void setPed(bool walkOn, bool dontWalkOn) {
  // walkOn -> ledVerdePedestre; dontWalkOn -> ledVermelhoPedestre
  digitalWrite(ledVerdePedestre,    walkOn      ? ON : OFF);
  digitalWrite(ledVermelhoPedestre, dontWalkOn  ? ON : OFF);
}

// Parse bits "b0..b4" para saídas
// Ordem: veh_g, veh_y, veh_r, ped_w, ped_dw
void applyOutputsFromBits(const String& bits) {
  if (bits.length() < 5) return;
  bool veh_g  = bits[0] == '1';
  bool veh_y  = bits[1] == '1';
  bool veh_r  = bits[2] == '1';
  bool ped_w  = bits[3] == '1';
  bool ped_dw = bits[4] == '1';

  setCar(veh_r, veh_y, veh_g);
  setPed(ped_w, ped_dw);
}

// Le RC522; em caso de sucesso, envia TAG,<uidhex>
void pollRFID() {
  if (!mfrc522.PICC_IsNewCardPresent()) return;
  if (!mfrc522.PICC_ReadCardSerial())   return;

  // Monta UID em hex contínuo (sem espaços), MAIÚSCULO
  String uidHex = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    byte b = mfrc522.uid.uidByte[i];
    if (b < 0x10) uidHex += '0';
    uidHex += String(b, HEX);
  }
  uidHex.toUpperCase();

  Serial.print("TAG,");
  Serial.println(uidHex);

  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
}

// Le botão analógico; em caso de pressão, envia BTN,PED (com debounce)
void pollButton() {
  int v = analogRead(pinoBotaoAnalogico);
  uint32_t now = millis();
  if (v > THRESH_BOTAO && (now - lastBtnMs) > DEBOUNCE_BTN_MS) {
    lastBtnMs = now;
    Serial.println("BTN,PED");
  }
}

// Trata linhas recebidas da serial
void handleLine(const String& line) {
  if (line.length() == 0) return;

  if (line.startsWith("OUT,")) {
    String bits = line.substring(4);
    bits.trim();
    applyOutputsFromBits(bits);
    return;
  }

  if (line == "PING") {
    Serial.println("PONG");
    return;
  }

  // Opcional: logging desconhecido
  // Serial.print("UNK: "); Serial.println(line);
}

void setup() {
  Serial.begin(115200);

  // LEDs
  pinMode(ledVermelhoCarro,    OUTPUT);
  pinMode(ledAmareloCarro,     OUTPUT);
  pinMode(ledVerdeCarro,       OUTPUT);
  pinMode(ledVermelhoPedestre, OUTPUT);
  pinMode(ledVerdePedestre,    OUTPUT);

  // Estado inicial seguro (carro vermelho + pedestre não caminhe)
  setCar(true, false, false);
  setPed(false, true);

  // SPI remapeado + RC522
  SPI.begin(RFID_SCK, RFID_MISO, RFID_MOSI, RFID_SS);
  mfrc522.PCD_Init();

  // ADC
  analogReadResolution(12); // ESP32: 0..4095
  // (opcional) analogSetAttenuation(ADC_11db);

  Serial.println("ESP32 pronto (bridge). Aguardando OUT/ PING do PC...");
}

void loop() {
  // ----- Recepção serial (linha por linha) -----
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      rxLine.trim();
      handleLine(rxLine);
      rxLine = "";
    } else {
      if (rxLine.length() < 100) rxLine += c; // proteção simples
    }
  }

  // ----- Entradas -----
  pollButton();
  pollRFID();

  // ----- Heartbeat -----
  uint32_t now = millis();
  if (now - lastHB >= HEARTBEAT_MS) {
    lastHB = now;
    Serial.println("HB");
  }

  // Loop “rápido” e não bloqueante
  delay(5);
}
