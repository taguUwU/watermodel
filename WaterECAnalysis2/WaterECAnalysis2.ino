#include "DFRobot_EC.h"
#include "DFRobot_ESP_PH.h"
#include <EEPROM.h>
#include <OneWire.h>
#include <DallasTemperature.h>

#define EC_PIN 36
#define PH_PIN 39
#define C_PIN 32
DFRobot_ESP_PH ph;

float V, EC, K, temperature, V_PH, PH;
DFRobot_EC ec;

OneWire oneWire(C_PIN);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(115200);
  ec.begin();
  sensors.begin();
  EEPROM.begin(32);
  ph.begin();
}

void loop() {
    static unsigned long timepoint = millis();
    if (millis() - timepoint > 3000U) {  // 每1秒讀取一次
        
        timepoint = millis();

        sensors.requestTemperatures();
        temperature = sensors.getTempCByIndex(0);

        V = analogRead(EC_PIN) / 4095.0 * 3.3; // ESP32 ADC 讀取轉換成 V
        K = 12.88/1.859; // K = EC(標準液) / V(標準液) 
        EC = K*V*1000;  // 轉換 EC 值

        V_PH = analogRead(PH_PIN) / 4095.0 * 3300; // ESP32 ADC 讀取轉換成 V
        PH = ph.readPH(V_PH, temperature);

        Serial.print(temperature, 1);
        Serial.println(" °C");
        Serial.print(EC, 2);
        Serial.println(" μs/cm");
        Serial.print(V, 2);
        Serial.println(" V");
        Serial.println(PH, 2);
    }
    //ec.calibration(V, temperature);
    //ph.calibration(V_PH, temperature);
}