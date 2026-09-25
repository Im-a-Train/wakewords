# Training «Hey Henä» (microWakeWord, lokal in Docker)

Diese Pipeline trainiert ein eigenes Wake Word für den **Home Assistant Voice PE**.
Sie läuft lokal in Docker, standardmässig auf einer **AMD-Grafikkarte über ROCm**
(getestet ist die Konfiguration für die Radeon RX 9070 XT / gfx1201).

Was sie besser macht als das Colab-Notebook:

| Problem bisher | Lösung hier |
| --- | --- |
| Englische TTS spricht «hene» als «hiin» | Stimmen sprechen die Lautschrift `hˈeɪ hˈɛnæ` direkt, mit Varianten |
| Nur ein Sprecher bzw. wenige Stimmen | ~1300 Sprecher aus 15 Piper-Modellen (DE, EN, LU, NL) |
| Keine echten Berndeutsch-Stimmen | Eigene Aufnahmen werden automatisch zerschnitten und stark gewichtet |
| Nur englische Negativbeispiele | Ähnlich klingende Wörter + eigene Berndeutsch-Aufnahmen als Negative |
| Schwellwert geraten | `evaluate` misst Erkennung und Fehlauslösungen/h und empfiehlt den Wert |

## Voraussetzungen

- **Windows:** WSL2 mit Ubuntu 24.04 und Docker, siehe [Abschnitt «Windows»](#windows-wsl2)
- **Linux:** Docker und Docker Compose, AMD-Treiber mit ROCm-Unterstützung auf dem Host
  (`/dev/kfd` und `/dev/dri` müssen existieren).
  Für die RX 9070 XT braucht es einen aktuellen Kernel/amdgpu-Treiber (ROCm ≥ 6.4.1).
  Prüfen auf dem Host: `ls /dev/kfd /dev/dri` und `groups` (du solltest in `video` und `render` sein).
- ca. 60 GB freier Speicher (ROCm-Image ~30 GB, Downloads ~8 GB, Features ~15 GB)
- mindestens 16 GB RAM, besser 32 GB

Das Standard-Image `rocm/tensorflow:rocm7.2.4-py3.12-tf2.20-dev` ist von AMD
ausdrücklich auch für `gfx1201` (RX 9070 / 9070 XT) gebaut.

## Windows (WSL2)

Unter Windows läuft alles in WSL2. ROCm spricht die Karte dort nicht über `/dev/kfd` an,
sondern über den Windows-Treiber (`/dev/dxg` + AMDs Bibliothek `librocdxg`). AMD
unterstützt das offiziell für die RX 9070 XT und TensorFlow 2.20 (ROCm 7.2).
`run.sh` erkennt WSL automatisch und nimmt `docker-compose.wsl.yml` dazu.

Einmalige Einrichtung:

1. **Windows:** AMD-Adrenalin-Treiber **26.2.2 oder neuer** installieren.
2. **WSL mit Ubuntu 24.04** installieren (PowerShell als Administrator):
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
3. **Im Ubuntu-Terminal: Docker Engine** installieren (<https://docs.docker.com/engine/install/ubuntu/>).
   Docker Desktop mit WSL-Integration kann auch gehen, aber die ROCm-Bibliotheken müssen
   im Ubuntu liegen. Mit Docker direkt in Ubuntu ist das sicher der Fall.
4. **librocdxg installieren** (verbindet ROCm mit dem Windows-Treiber), gemäss
   <https://github.com/ROCm/librocdxg> (fertiges `.deb` oder aus dem Quellcode). Danach muss
   `/opt/rocm/lib/librocdxg.so` existieren. Liegt die Datei woanders, den Pfad setzen:
   `export ROCDXG_LIB=/pfad/librocdxg.so`.
5. **Repo ins Linux-Dateisystem klonen**, nicht nach `/mnt/c/...`. Unter `/mnt/c` sind die
   Dateizugriffe extrem langsam, und das Training liest viele GB:
   ```bash
   cd ~ && git clone https://github.com/Im-a-Train/wakewords.git
   cd wakewords/training
   ```
6. Weiter mit dem Schnellstart unten. `./run.sh check-gpu` muss die GPU zeigen.

Aufnahmen vom Handy kopierst du im Windows-Explorer nach
`\\wsl$\Ubuntu-24.04\home\<benutzer>\wakewords\training\recordings\positive\<name>\`.
Die fertigen Modelle findest du dort unter `...\training\output\`.

Falls die GPU unter WSL nicht erkannt wird:

- Unter Windows den Adrenalin-Treiber aktualisieren und `wsl --update` ausführen.
- Im Ubuntu prüfen: `ls -l /dev/dxg /usr/lib/wsl/lib/libdxcore.so /opt/rocm/lib/librocdxg.so`.
- Ein neueres ROCm-Image probieren (ab ROCm 7.13 braucht es `HSA_ENABLE_DXG_DETECTION` nicht mehr):
  `docker compose build --build-arg BASE_IMAGE=rocm/tensorflow:rocm7.14.1-ubuntu24.04-py3.12-tf2.20`
- Notfalls auf der CPU trainieren: `WWTRAIN_CPU=1 ./run.sh train`.

## Schnellstart

```bash
cd training

# 1. Image bauen und GPU prüfen
docker compose build
./run.sh check-gpu          # sollte «GPUs: [PhysicalDevice(... GPU:0 ...)]» zeigen

# 2. Daten herunterladen (einmalig, ~8 GB)
./run.sh download

# 3. Aussprache prüfen: Beispiele in output/preview/ anhören
./run.sh preview

# 4. Eigene Aufnahmen ablegen (siehe unten), dann alles durchlaufen lassen
./run.sh all
```

Am Ende liegen in `output/`:

- `hey_hena.tflite` – das Modell
- `hey_hena.json` – das Manifest für ESPHome (mit empfohlenem Schwellwert)
- `hey_hena_evaluation.txt` – Messergebnisse

Einzelne Schritte kannst du auch separat starten:
`import`, `tts`, `features`, `train`, `evaluate`.

## Eigene Aufnahmen (das Wichtigste!)

Synthetische Stimmen können kein Berndeutsch. Echte Aufnahmen von allen, die
den Assistenten nutzen werden, machen den grössten Unterschied.

### Positiv: «Hey Henä»

Für jede Person einen Ordner anlegen:

```
recordings/positive/melvin/aufnahme1.m4a
recordings/positive/melvin/kueche_leise.m4a
recordings/positive/anna/sprachmemo.mp3
```

- Mit dem Handy (Sprachmemo) oder am PC aufnehmen, das Format spielt keine Rolle.
- In einer Aufnahme **viele Male «Hey Henä»** sagen, dazwischen **1–2 Sekunden Pause**.
  Die Datei wird automatisch in einzelne Schnipsel zerschnitten.
- Ziel: **50–100 Mal pro Person**, verteilt auf mehrere Aufnahmen:
  - normal, leise, laut, schnell, langsam, fragend, müde, beim Nebenbei-Reden
  - nah am Handy und aus 2–4 m Distanz
  - in verschiedenen Räumen (Küche, Wohnzimmer)
  - wenn möglich auch mit Nebengeräuschen (Radio, Dampfabzug)
- Kinder und alle Erwachsenen im Haushalt mitmachen lassen.

Nach `./run.sh import` liegen die Schnipsel in `data/clips/real_positive/train` und
`.../test`. **Kurz reinhören** und falsch geschnittene Schnipsel löschen. Rund 20 %
der Schnipsel werden nie trainiert und dienen nur dem ehrlichen Test.

### Negativ: alles ausser «Hey Henä»

Lange Aufnahmen in `recordings/negative/` legen: Gespräche am Esstisch, Radio SRF 1,
Fernsehen, Telefonate. 1–3 Stunden sind super. Diese Aufnahmen werden als
Negativbeispiele und als Hintergrundgeräusch verwendet. Zudem misst `evaluate` damit
die Fehlauslösungen pro Stunde.

## Aussprache anpassen

In `config.yaml` unter `tts.positive.phrases` steht die Lautschrift (IPA). Mit
`./run.sh preview` erzeugst du Beispiele in `output/preview/positive/` zum Anhören.
Klingt es nicht wie ihr, dann passe die Lautschrift an:

| Laut | Zeichen | Beispiel |
| --- | --- | --- |
| e wie in «Bett» | `ɛ` | `hˈɛnæ` |
| offenes ä (Berndeutsch Endung) | `æ` | `hˈɛnæ` |
| langes ä | `ɛː` | `hˈɛːnæ` |
| Hey (englisch) | `heɪ` | |
| Hey (breiter) | `hɛɪ`, `hæɪ` | |
| Betonung | `ˈ` vor der Silbe | |

Unter `tts.negative.phrases` stehen ähnlich klingende Wörter, auf die das Modell
**nicht** reagieren soll. Namen, die bei euch oft fallen, dort ergänzen (z. B. andere
Familienmitglieder, Haustiere).

## Weil «zu wenig oft auslösen» das Problem war

Die Standardwerte sind bewusst auf eine hohe Erkennungsrate ausgelegt:

- echte Aufnahmen werden 15× (jedes Mal anders verrauscht und verhallt) verwendet
- `training.negative_class_weight` startet mit 15 statt 20
- die Validierung, die die «besten» Gewichte auswählt, nutzt eure echten Stimmen

Falls es danach immer noch zu selten reagiert:

1. mehr eigene Aufnahmen (vor allem aus Distanz und leise)
2. in `evaluate` den tieferen Schwellwert wählen, `probability_cutoff` im Manifest senken
3. `training.negative_class_weight` weiter senken (z. B. `[10, 15]`)
4. `sliding_window_size` im Manifest von 5 auf 3 senken

Falls es zu oft fälschlich auslöst: das Gegenteil, und die auslösenden Situationen
aufnehmen und in `recordings/negative/` legen.

## Alte Modelle vergleichen

Das ganze Repo ist im Container unter `/repo` eingebunden:

```bash
./run.sh evaluate --model /repo/hey_hene_4.tflite
./run.sh evaluate                                 # das neue Modell
```

So siehst du schwarz auf weiss, ob das neue Modell auf euren Stimmen besser ist.

## Ins Voice PE bringen

1. `output/hey_hena.tflite` und `output/hey_hena.json` ins Repo-Root kopieren, committen, pushen.
2. In `my-wakeword.yaml` das Modell eintragen:

   ```yaml
   micro_wake_word:
     models:
       - model: https://raw.githubusercontent.com/Im-a-Train/wakewords/refs/heads/main/hey_hena.json
         id: hey_hena
   ```

3. In ESPHome neu kompilieren und flashen.

Falls das Gerät meldet, dass das Modell nicht geladen werden kann, `tensor_arena_size` in
`config.yaml` erhöhen (z. B. 35000) und `./run.sh evaluate` erneut ausführen, damit das
Manifest neu geschrieben wird.

## GPU-Probleme

- `check-gpu` findet keine GPU:
  - Prüfen, ob `/dev/kfd` existiert und der Benutzer in `video`/`render` ist.
  - In `docker-compose.yml` `HSA_OVERRIDE_GFX_VERSION: "12.0.1"` aktivieren.
  - Ein anderes ROCm-Image probieren, z. B. neueres ROCm:
    `docker compose build --build-arg BASE_IMAGE=rocm/tensorflow:rocm7.14.1-ubuntu24.04-py3.12-tf2.20`
    (Tags: <https://hub.docker.com/r/rocm/tensorflow/tags>)
- Notfalls läuft alles auf der CPU: `WWTRAIN_CPU=1 ./run.sh train`. Das ist deutlich
  langsamer (Stunden statt Minuten), funktioniert aber gleich.

## Aufbau

```
training/
├── config.yaml          alle Einstellungen
├── run.sh               Einstieg (docker compose run)
├── Dockerfile           ROCm- oder CPU-Image
├── docker-compose.yml
├── docker-compose.wsl.yml   Ergänzung für Windows/WSL2 (automatisch)
├── wwtrain/             Python-Pipeline
│   ├── download.py      Stimmen, Hall (MIT RIR), Geräusche (AudioSet, FMA), Negativ-Features
│   ├── recordings.py    eigene Aufnahmen importieren und schneiden
│   ├── tts.py           Piper-TTS mit Lautschrift
│   ├── features.py      Augmentierung und Spektrogramme
│   ├── train.py         microWakeWord-Training und Manifest
│   └── evaluate.py      Streaming-Test wie auf dem Gerät
├── recordings/          eure Aufnahmen (nicht im Git)
├── data/                Downloads, Features, Checkpoints (nicht im Git)
└── output/              fertiges Modell und Berichte
```

Hinweis zu Lizenzen: Die heruntergeladenen Hintergrund- und Negativdaten haben
unterschiedliche Lizenzen. Ein so trainiertes Modell ist nur für die private,
nicht-kommerzielle Nutzung gedacht.
