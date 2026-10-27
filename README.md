# モーション × BER 同時計測

OptiTrack（Motive / NatNet）で身体の 7 マーカーを撮りながら、ADP3250 × 2 台の NFC VHBR 上り BER を測る。
画面にはマーカーの 3D 表示と Web カメラの映像も出し、BER の記録中はカメラも録画する。
BER の 1 収録ごと・テストフレーム 1 枚ごとに、そのときの姿勢を対応付けて残す。

先々は、学習済みモデルでモーションからコイルのインピーダンスを推定して、リーダの W2−W1 位相差を
自動で合わせる。そのための差し込み口（`--phase-model`）だけ用意してあり、**いまは何もしない**。

## 何をどこから使っているか

元の 2 つは**書き換えていない**。こちらは読みに行くだけ。

| | 使うもの | どう使うか |
|---|---|---|
| `../2026_nfc_vhbr_ber` | `ber_gui.py` 一式（装置の操作、受信部、画面） | そのまま import して 4 か所だけ差し替える（下記） |
| `../nanoVNA_OptiTrack_get_dataset` | `NatNetSDK/Samples/PythonClient/NatNetClient.py` | そのまま使う |
| 〃 | `sync_optitrack_nanovna.py` のラベル認識 | **`motion.py` に写した**（あちらは tkinter・pyserial・scikit-rf を import するので BER 側の環境から読めない） |

場所が違うときは環境変数 `NFC_BER_DIR` / `NATNET_DIR` で指定する。

差し替えている所（`motion_ber_gui.py`）:

| 差し替え | 目的 |
|---|---|
| `bench.Bridge.record` | 収録を始めた時刻を `perf_counter` で控える |
| `fast_demod.FastReceiver.process` | 直近の復調結果（各フレームの位置・誤り数）を控える |
| `ber_gui.SessionLog` | 記録フォルダにモーションの CSV を足す |
| `ber_gui.Handler` / `live_worker` | 画面にモーションの欄を足す、終了時に記録を閉じる |

**`ber_gui.py` の中でこれらの名前や呼び方を変えたら `motion_ber_gui.py` も直すこと。**
画面は `ber_gui.html` の `<div class="cols">` の前に欄を差し込んでいる。無ければ起動時に分かるエラーになる。

## 使い方

```bash
uv run motion_ber_gui.py                       # http://127.0.0.1:8766 が開く
uv run motion_ber_gui.py --connect             # 起動と同時に OptiTrack (127.0.0.1) へ繋ぐ
uv run motion_ber_gui.py --group lower         # 下半身
uv run motion_ber_gui.py --fake-motion         # Motive 無し。合成モーションで画面と記録を試す
uv run motion_ber_gui.py --camera 0            # カメラ 0 も開く（画面の「カメラ」欄からも開ける）
uv run python tests/test_motion_ber.py         # 装置無しで確かめられる部分のテスト
```

1. Motive を起動して Data Streaming を ON（Labeled Markers を含める。既定は 127.0.0.1・Unicast）。
2. 画面の「モーション」欄で計測部位を選んで「接続」。緑の ● が 7 個そろうのを確かめる。
   カメラも使うなら「カメラ」欄で番号を入れて「接続」（この PC では 1 = 外付けの C920、0 = 内蔵。画面の既定は 1）。
3. あとは `ber_gui.py` と同じ。「開始」で BER の測定と記録が始まる。

- **OptiTrack のフレームが来ていないと「開始」を断る。** BER だけ測りたいときは「OptiTrack なしでも開始できる」を入れる。
- 測定中は計測部位の変更・接続し直し、カメラの繋ぎ直しができない（CSV の列や録画が途中で変わるため）。
- 起動のオプション（`--div`、`--reader`、`--tag`、`--payload`）と装置の注意は `../2026_nfc_vhbr_ber/README.md` と同じ。
  **強制終了しないこと**も同じ（画面の終了ボタンか `POST /api/shutdown`）。
- Motive が別の PC のときは「Motive の IP」と「この PC の IP」を入れる。Multicast なら チェックを入れる。

### 3D 表示とカメラ

- **3D 表示**: 最新の姿勢を約 30 Hz で描く。ドラッグで回転、ホイールで拡大縮小、ダブルクリックで戻す。上の軸は Motive の設定に合わせて選ぶ（この台は Z が上、既定も Z）。ボーン・配色・肘／膝の角度は `live_dashboard.py` と同じ。見えていないマーカーは赤字で出る。
- **カメラ**: OpenCV（DirectShow）で開き、640 px 幅の MJPEG で画面に流す。「カメラも録画」が入っていれば、BER の記録と同じ期間だけ記録フォルダに録画する（元の大きさのまま）。
- OpenCV が入っていない環境でも BER とモーションは使える（カメラ欄が無効になるだけ）。

### マーカー名の対応

ラベル付きマーカーの `marker_id = i` を、部位ごとの名前リストの `i` 番目に割り当てる（`sync_optitrack_nanovna.py` と同じ規則）。
並びは `motion.py` の `MARKER_GROUPS`。Motive のアセットを変えたら、画面の「受信フレームの構造」で `marker=` を見て合わせる。
上半身と下半身のアセットを同時に配信しているときは `MARKER_MODEL_ID` に `model=` の値を入れる。

## 記録されるもの

`ber_gui.py` の記録フォルダ `~/nfc_vhbr_captures/log/<日時>_<条件>/` に、元の `conditions.json`・`ber_log.csv` と並べて:

| ファイル | 1 行 | 主な列 |
|---|---|---|
| `motion.csv` | Motive の 1 フレーム（全部） | `t`、`wall`、`OptiFrame`、`OptiTime`、`OptiMidExp`、`n_valid`、`<マーカー>_X/Y/Z` [m]（見えていなければ空） |
| `ber_motion.csv` | BER の 1 収録（既定 0.1 秒） | `t_start`/`t_end`、レベル・位相（`phase_src` = manual/auto）、BER、収録窓の `motion_frames`・`valid_frac`（全点見えていた割合）、`<マーカー>_X/Y/Z`（窓の平均）、`<マーカー>_dev`（窓の中で平均から最も離れた距離 [m]）、`raw_file`（この収録の生データ。無ければ空） |
| `cap_*.ad3` | 収録 1 回の生データ（「生データも記録」のときだけ。1 本約 20 MB） | 「記録の再生」で波形を描くのに使う |
| `frames.csv` | テストフレーム 1 枚（600 ビット） | `cap`、`t`、`errors`（−1 = 取得失敗）、`OptiFrame`（いちばん近い Motive フレーム） |
| `camera.mp4` / `camera_frames.csv` | カメラの 1 コマ | `video_frame`（動画の何コマ目か）、`t`（そのコマを受けた時刻）。カメラを繋いでいて「カメラも録画」のときだけ |
| `motion_conditions.json` | — | 部位、マーカー名、カメラの設定、時計の対応（`perf0` ↔ `wall0`） |

`camera.mp4` の再生速度は名目の fps なので、**動画の時刻ではなく `camera_frames.csv` の `t` を使うこと**（カメラの実際のコマ間隔は明るさなどで変わる）。

`t` はどの CSV でも同じ時計（記録開始からの秒）なので、`frames.csv` と `motion.csv` は `t` か `OptiFrame` で結合できる。
学習用には、フレームの誤りを `OptiFrame` で姿勢に結び付けるのが最も細かい。

測定は**断続的**: 0.1 秒撮って約 0.5 秒かけて USB で吸い上げる、の繰り返し（`ber_gui.py` の README「速度」）。
BER があるのは収録窓の中だけで、モーションは全区間ある。

## 同期のしかた

ADP3250 と Motive は外部トリガで結ばれていない。どちらも PC の `time.perf_counter()` で時刻を打って突き合わせる。

| | 時刻 | 乗っている未較正の遅れ |
|---|---|---|
| 収録窓 | `FDwfAnalogInConfigure` を呼ぶ直前 〜 + 点数 / fs | 開始を USB で伝える遅れ（~1 ms の見込み） |
| テストフレーム | 収録窓の先頭 + 窓の位置 / 受信部の fs | 上と同じ（DDC の遅れ < 1 µs は無視） |
| モーション | NatNet のフレームを受けた時刻 | Motive の処理・配信（数 ms） |

フレーム 1 枚は FC/8 で約 0.4 ms、モーションは 120 fps で 8.3 ms 間隔なので、精度はモーション側で決まる。
ずれをきちんと詰めたいときは、両方に出る物理的な出来事（コイルを急に動かして BER が跳ねる所）で一度較正する。

**装置を繋いだ状態ではまだ動かしていない**（この PC には WaveForms も Motive も無い）。
装置無しのテストで確かめたのは: マーカーの対応付け、NatNet のコールバック、収録窓とモーションの突き合わせ、
3 つの CSV、位相の差し込み口、HTTP の各 API、画面の表示（合成モーション）、カメラの配信と録画（合成カメラと内蔵カメラ）。
初回は `--fake-motion` → Motive だけ → ADP3250 も、の順に確かめること。

## 記録の再生

「記録の再生」欄で記録フォルダ（`~/nfc_vhbr_captures/log/` の中の、`motion.csv` があるもの）を選んで「開く」。
シークバー 1 本で、次が同じ時刻にそろって動く。

| | 何を出すか |
|---|---|
| 3D 表示 | `motion.csv` の、その時刻にいちばん近い姿勢 |
| カメラ | `camera.mp4` の、その時刻までに撮れた最後のコマ（`camera_frames.csv` の時刻で引く） |
| タイムライン | 収録ごとの BER（青 = 生データあり、灰 = なし）。クリックでその時刻へ |
| ①〜③（元の画面） | **生データのある収録の中だけ**、その時刻のテストフレームをその場で復調して描く（元の「再生」と同じ手順: `fd.ddc` → `receiver_for` → `make_view`） |

- 操作: 再生／停止、速さ（×0.02〜×2）、「収録の中だけ」（収録と収録の間＝転送待ち約 0.5 秒を飛ばす）、収録単位・テストフレーム単位の送り。
- **波形まで見たいときは、測定のときにライブ欄の「生データも記録」を入れる。** 0.1 秒の収録 1 回で約 20 MB（1 分で約 1.2 GB）。
  どの収録がどの `.ad3` かは `ber_motion.csv` の `raw_file` 列に残る。
- 1 本目の収録を開くときだけ、復調の下ごしらえ（`<名前>.bb8.npy` を作る）で 0.2 秒ほどかかる。以後は 1 フレーム 50 ms 前後。
- 再生中は BER の測定を始められない（閉じてから）。

## 位相の自動調整（先々用）

```bash
uv run motion_ber_gui.py --phase-model phase_model_example.py
```

渡したファイルの `predict_phase(motion) -> float | None` を**収録 1 回ごとに 1 回**呼び、返した位相を
GUI で入れたのと同じ経路で次の収録の前に反映する（`Bridge.set_phase`。続けて叩くとリーダが固まるので、この頻度より上げない）。
`motion` の中身と守ることは `phase_control.py`、モデルの書き方の例は `phase_model_example.py`。
画面の「位相の自動調整」で入り切りする。モデルが例外を出したら自動で切る。

位相の**指定値と実測には +3.9° のずれ**がある（175.53° 指定で実測 180.04°）。モデルが実測の位相を出すなら
`phase_control.PHASE_OFFSET_DEG` を足して指定値に直す。

## ファイル

| | |
|---|---|
| `motion_ber_gui.py` | 本体: 差し替え、記録（`MotionSessionLog`）、画面の欄、HTTP |
| `motion.py` | NatNet の受信、ラベル認識、リングバッファ、合成モーション、3D 表示の配色・ボーン |
| `replay.py` | 記録フォルダを読み、時刻で姿勢・カメラのコマ・収録とフレームを引く（記録の再生） |
| `camera.py` | Web カメラ: 画面への MJPEG、記録フォルダへの録画（コマごとの時刻つき） |
| `phase_control.py` / `phase_model_example.py` | 位相の自動調整の差し込み口とひな形 |
| `tests/test_motion_ber.py` | 装置無しのテスト |
