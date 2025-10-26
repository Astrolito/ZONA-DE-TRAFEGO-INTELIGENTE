// -----------------------------
// Semáforo carro x pedestre (ESP32)
// - LEDs ativos em nível BAIXO (LOW = ligado)
// - Botão analógico (10s de espera)
// - Leitor RFID RC522 via SPI (5s de espera)
// -----------------------------

#include <SPI.h>
#include <MFRC522.h>

// ---------- Pinos LEDs (mantidos) ----------
const int ledVermelhoCarro    = 18;
const int ledAmareloCarro     = 19;
const int ledVerdeCarro       = 21;
const int ledVermelhoPedestre = 22;
const int ledVerdePedestre    = 23;

// ---------- Botão analógico ----------
const int pinoBotaoAnalogico  = 34;   // ADC somente entrada

// ---------- SPI do RC522 (remapeado p/ não conflitar com LEDs) ----------
const int RFID_SS   = 25;  // SDA/SS do RC522
const int RFID_RST  = 4;   // RST do RC522
const int RFID_SCK  = 14;  // SCK
const int RFID_MISO = 27;  // MISO
const int RFID_MOSI = 26;  // MOSI

MFRC522 mfrc522(RFID_SS, RFID_RST);

// ---------- Mapa lógico ----------
#define ON   LOW
#define OFF  HIGH

// ---------- Tempos ----------
const int THRESH_BOTAO        = 100;     // limiar do ADC para considerar "pressionado"
const int T_VERIFICA_LOOP_MS  = 50;      // período do loop (mais responsivo)
const int T_AMARELO_MS        = 3000;
const int T_PISCA_AMARELO_ON  = 500;
const int T_PISCA_AMARELO_OFF = 500;
const int N_PISCAS_AMARELO    = 5;
const int T_ALL_RED_GAP_MS    = 2000;
const int T_PED_VERDE_MS      = 10000;
const int T_PISCA_PED_ON      = 500;
const int T_PISCA_PED_OFF     = 500;
const int N_PISCAS_PED        = 4;
const int T_FECHAR_PED_MS     = 3000;

// Esperas após acionamento
const int T_ESPERA_BOTAO_MS   = 10000;   // 10s com botão
const int T_ESPERA_RFID_MS    = 5000;    // 5s com RFID

// ---------- Estado ----------
int  pedidoTravessia = 0;
bool acionadoPorRFID = false;

// ---------- Utilidades ----------
void setCar(bool redOn, bool yellowOn, bool greenOn) {
  digitalWrite(ledVermelhoCarro,  redOn    ? ON : OFF);
  digitalWrite(ledAmareloCarro,   yellowOn ? ON : OFF);
  digitalWrite(ledVerdeCarro,     greenOn  ? ON : OFF);
}

void setPed(bool redOn, bool greenOn) {
  digitalWrite(ledVermelhoPedestre, redOn   ? ON : OFF);
  digitalWrite(ledVerdePedestre,    greenOn ? ON : OFF);
}

void blink(int pin, int vezes, int tOnMs, int tOffMs) {
  for (int i = 0; i < vezes; i++) {
    digitalWrite(pin, ON);
    delay(tOnMs);
    digitalWrite(pin, OFF);
    delay(tOffMs);
  }
}

// Lê cartão; retorna true se houve leitura válida e imprime UID
bool checkRFID() {
  if (!mfrc522.PICC_IsNewCardPresent()) return false;
  if (!mfrc522.PICC_ReadCardSerial())   return false;

  // Mostra UID
  Serial.print("RFID UID: ");
  for (byte i = 0; i < mfrc522.uid.size; i++) {
    Serial.print(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " ");
    Serial.print(mfrc522.uid.uidByte[i], HEX);
  }
  Serial.println();

  // Para a comunicação com o cartão atual
  mfrc522.PICC_HaltA();
  mfrc522.PCD_StopCrypto1();
  return true;
}

void setup() {
  Serial.begin(115200);

  // LEDs
  pinMode(ledVermelhoCarro,    OUTPUT);
  pinMode(ledAmareloCarro,     OUTPUT);
  pinMode(ledVerdeCarro,       OUTPUT);
  pinMode(ledVermelhoPedestre, OUTPUT);
  pinMode(ledVerdePedestre,    OUTPUT);

  // SPI remapeado
  SPI.begin(RFID_SCK, RFID_MISO, RFID_MOSI, RFID_SS);
  mfrc522.PCD_Init(); // usa SPI padrão já inicializado acima com os pinos remapeados

  // Estado inicial: carro verde, pedestre vermelho
  setCar(false, false, true);
  setPed(true, false);

  Serial.println("Sistema pronto. Aproxime um cartão/tag (RFID) ou acione o botao analogico.");
}

void loop() {
  // --- Entrada: botão analógico ---
  int valorAnalogico = analogRead(pinoBotaoAnalogico);
  if (!pedidoTravessia && valorAnalogico > THRESH_BOTAO) {
    pedidoTravessia = 1;
    acionadoPorRFID = false;
    Serial.println("Botao acionado: aguardando 10 segundos...");
  }

  // --- Entrada: RFID ---
  if (!pedidoTravessia && checkRFID()) {
    pedidoTravessia = 1;
    acionadoPorRFID = true;
    Serial.println("RFID acionado: aguardando 5 segundos...");
  }

  if (pedidoTravessia == 1) {
    // Espera condicionada ao tipo de acionador
    if (acionadoPorRFID) {
      delay(T_ESPERA_RFID_MS);
    } else {
      delay(T_ESPERA_BOTAO_MS);
    }

    // 1) Amarelo
    delay(3000);                    // buffer (mantido do seu fluxo original)
    setCar(false, true, false);     // amarelo ON
    delay(T_AMARELO_MS);

    // 2) Pisca amarelo
    digitalWrite(ledAmareloCarro, OFF);
    blink(ledAmareloCarro, N_PISCAS_AMARELO, T_PISCA_AMARELO_ON, T_PISCA_AMARELO_OFF);
    digitalWrite(ledAmareloCarro, OFF);

    // 3) Carro vermelho + gap
    setCar(true, false, false);
    delay(T_ALL_RED_GAP_MS);

    // 4) Pedestre verde
    setPed(false, true);
    delay(T_PED_VERDE_MS);

    // 5) Pisca verde pedestre
    digitalWrite(ledVerdePedestre, OFF);
    blink(ledVerdePedestre, N_PISCAS_PED, T_PISCA_PED_ON, T_PISCA_PED_OFF);

    // 6) Fecha pedestre e reabre carro
    setPed(true, false);
    delay(T_FECHAR_PED_MS);
    setCar(false, false, true);

    // Libera novo ciclo
    pedidoTravessia = 0;
    acionadoPorRFID = false;
  }

  delay(T_VERIFICA_LOOP_MS);
}
