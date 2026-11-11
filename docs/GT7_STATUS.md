# GT7 su Apple Silicon: stato e prossimi passi

Aggiornato: 2026-11-11. Pubblicato come `main` su questo repo privato.

Gran Turismo 7 (PS4, CUSA24769 v1.00) gira sul runtime di Driveclub-Port con il renderer GCN→Metal.
Questo repo contiene solo codice: nessun file del gioco, chiave o cattura.

## Dove siamo

- Avvio, menu, impostazioni display e Music Rally fino alla gara funzionano. La schermata
  "Display Settings" con la strada nel bosco coincide quasi al pixel con il video PS5 di riferimento.
- Gara in abitacolo: prato, asfalto, cartelloni, HUD, specchietto corretti. ~18-20 fps su M1 Max.

### Lavoro del 2026-11-09 / 11

| Commit | Cosa |
|---|---|
| `358ba66` | **Filmati**: libSceAvPlayer su AVFoundation (`psstudio.mp4` all'avvio con video e audio 6 canali). Due bug del renderer trovati così: texture USCALED lette come UNORM (ora ×255/65535 dopo la lettura) e provenienza degli SGPR che attraversava gli `S_BRANCH` (un'etichetta raggiunta solo da salti ora prende lo stato del salto; A/B `DC_NO_JUMP_PROVENANCE=1`) |
| `3e6219a` | Compilazione shader: un worker per P-core, prewarm a QoS normale fino alla prima gara: prewarm a freddo 453 s (prima oltre 20 min) |
| `12cab7c` | L'Arcade ricorda pista, meteo e auto (script Adhoc riscritti al caricamento) |
| `adc0ed7`, `119c6f6`, `0bde551` | Fix Scapes portati da shadgt7 (FIX-GT7-126/127/130/131/133/134/135): **da provare in Scapes** |
| `a92fdff`, `bb7f458` | Server asset locale: concessionaria Leggendarie offline tramite il client HTTPS del gioco (**da verificare nella UI**) |
| `f1a5d8d` | Decifratura dei file del gioco 4 blocchi ChaCha alla volta: CPU per caricare una gara 7,9 → 3,7 s |
| `88bf079`, `4ee49c5` | Tessellation: patch e slice LDS come uniform (opt-in `DC_TESS_UNIFORMS=1`) |

### Lavoro del 2026-11-06 / 07

| Commit | Cosa |
|---|---|
| `6251306`, `b652b68`, `fbb0683`, `9d69102` | Cattura di un frame GPU e replay offline (`DC_CAPTURE*`, `DC_FRAME_REPLAY`), pad registrato ancorato agli eventi del log |
| `6273f14` | Replay: si assesta finché nessuna ring avanza prima di ogni batch di memoria; il frame 1 è identico bit per bit tra run |
| `777d784` | Avviso quando una texture chiede più di 2048 layer (`DC_NO_LAYER_CLAMP=1` per A/B) |
| `4fe145b` | Tessellation attiva di default (`DC_NO_TESS=1` o `DC_TESS=0` per spegnerla): compare il corpo dello specchietto |
| `e3427af` | Emulazione delle op F64 (Metal non ha i double) con soft-float su interi, testata contro double veri |
| `9539273` | Le texture PRT (tile mode 16) mappate una pagina da 64 KB alla volta ora si caricano: **prato verde, asfalto con texture, materiali dell'abitacolo** (prima erano dummy 1x1) |
| `afa2597` | Le texture PRT con pagine non residenti leggono i buchi come zero (`DC_NO_SPARSE_TEX=1` per A/B) |
| `6a669cd` | Esperimento `DC_UNTRACKED=1` (niente hazard tracking sui buffer della memoria guest): nessun guadagno, frame corrotti; resta solo come interruttore di debug |

## Cosa fare alla ripresa

### 1. Prestazioni: il limite è la CPU (priorità)

Misure in gara:

- `DC_UNTRACKED=1` non cambia gli fps (19,2 contro 19,6-20,0): la serializzazione dei pass GPU non è il limite.
- Profilo parziale con `DC_GPU_STATS=1 DC_CPU_PROFILE=1`: il thread CP usa **43 ms di CPU per frame**
  su 51,5 ms di frame. Dentro i draw 31,8 ms (risorse 15,7, funzioni 7,9, resolve 11,1,
  `GetTexture` 6,8 ms con 6110 chiamate e 5142 hit nello stesso frame, `LinearTable` 2,3 ms),
  fuori dai draw 19,8 ms (18,3 non attribuiti). Campione forse precedente alla gara: da ripetere.

Passi:

1. Ripetere il profilo fino in gara, a macchina scarica (senza asitop/powermetrics):
   `DC_GPU_STATS=1 DC_CPU_PROFILE=1 DC_DUMP_FRAMES=<dir> DC_DUMP_EVERY=500 ./run-gt7-replay.sh keys/anchored2-20261106.txt`
2. Profilare il thread CP con Instruments Time Profiler o `sample` per attribuire i 18 ms fuori dai draw.
3. Ridurre le sezioni "resources" e "functions" del draw: cache dell'oggetto texture per (T#, epoch),
   hit rate della Resolve cache (73%), meno hashing delle texture.
4. Dopo la CPU, i punti GPU dell'analisi: scratch del GS fuso fuori dall'arena da 16 MB
   (ogni wrap svuota la GPU), raggruppare i draw tessellati (12 split di pass per frame).

### 2. Grafica rispetto al PS5

- **Alberi a blob:** sono LOD lontani istanziati scelti dal gioco (draw diretti, istanze decise
  dalla CPU), non errori di rendering. Servirebbe un riferimento PS4 per sapere se è normale;
  il video di riferimento è PS5. Ipotesi: il LOD segue la risoluzione dinamica o il tempo di frame,
  quindi potrebbe migliorare con le prestazioni. Testato senza effetto: orologio GPU rallentato.
- Campo di colza, tende e spettatori del menu Music Rally mancano.
- Abitacolo più scuro e verdastro, riflesso del sole troppo forte, cielo più saturo.
- **Tinta blu** intermittente da una partita all'altra (indipendente da tessellation e F64):
  sospetta race tra code GPU legata a uno shader compute asincrono (`cs 0x9000fc9a00`).
- Un quadrilatero grigio visto una volta in alto a sinistra (flip 4200), non riprodotto.

### 3. Note operative

- Il loader a volte si blocca su "INITIALIZING" (circa 1 partita su 3): rilanciare.
- Il pad registrato può andare fuori sincrono alle impostazioni display: controllare che la
  partita arrivi in gara prima di usare le misure.
- Il replay va usato con `DC_INDIRECT_SYNC=1` e ogni confronto va ripetuto due volte;
  lo stato "abitacolo chiaro" che a volte esce nel replay è un artefatto del replay.
- Il replay non misura il throughput (fa ~26 attese GPU per frame di suo): le prestazioni
  si misurano in partita.
- Strumenti utili: `DC_PROBE='*:any:x,y'` (shader che scrivono un pixel, lento),
  `DC_FIND_ADDR` (V# che coprono un indirizzo, con `DC_TRACE_FRAME`), `DC_SKIP_PS`,
  `DC_LOG_RESTORED`, `DC_REPLAY_TARGETS_ONLY`, `DC_TESS_STATS`, `DC_REPLAY_LOG_KEPT`.
- README, app e percorsi ora sono di GT7 (2026-11-09): cache e salvataggi in `GT7Port`, log in `GT7 Logs`.
  Prima di renderlo pubblico: licenza da rivedere.
