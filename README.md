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
uv run motion_ber_gui.py --replay              # 記録の再生専用の画面（いちばん新しい記録。装置には触らない）
uv run motion_ber_gui.py --replay <フォルダ名>  # 記録を指定して再生
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
| ①〜③（元の画面） | **生データのある収録の中だけ**、その収録をライブと同じ手順（`fast_ddc` → 受信部の設定 → 窓の位置 → まとめて復調）で復調し、その時刻のテストフレームを描く。**フレームごとの誤り数はライブの `frames.csv` と一致する**（実機の記録で 406/406）。元の「再生」の手順（`ddc()` + 間引き 8 固定）は FC/8 で 1 枚もフレームが取れないので使っていない |

- 操作: 再生／停止、速さ（×0.02〜×2）、「収録の中だけ」（収録と収録の間＝転送待ち約 0.5 秒を飛ばす）、収録単位・テストフレーム単位の送り。
- **波形まで見たいときは、測定のときにライブ欄の「生データも記録」を入れる。** 0.1 秒の収録 1 回で約 20 MB（1 分で約 1.2 GB）。
  どの収録がどの `.ad3` かは `ber_motion.csv` の `raw_file` 列に残る。
- 最初の 1 回だけ 1.4 秒ほどかかる（受信部の立ち上げ）。収録を移るたびに 0.15 秒ほど（収録 1 本の復調）、同じ収録の中では 1 フレーム 30 ms 前後。
- 再生中は BER の測定を始められない（閉じてから）。

### 再生専用の画面（`--replay`）

`uv run motion_ber_gui.py --replay [フォルダ名]` で、再生だけの画面が http://127.0.0.1:8767 に開く
（測定用の GUI と同時に開ける。ADP3250・電源・OptiTrack・カメラには触らない）。窓の大きさに合わせて 1 画面に収まる。

| 左 | 右 |
|---|---|
| 動画に**実測したマーカーの骨格**を重ねたもの。下に再生の操作とタイムライン | 上: ① リーダ出力と包絡線。上に送ったビット、下に復調したビット（誤りは赤）<br>下: ③ 復調後の IQ と ④ BER の数値（現在・累積・この収録・このフレーム） |

**動画への重ね方:** カメラと OptiTrack は較正していないので、記録そのものから求める（`overlay.py`）。

1. 動画を 150 コマ間引いて MediaPipe Pose で骨格を取る（初回だけ約 10 秒。`pose_mediapipe.npz` に残す）
2. マーカーを MediaPipe の点に対応させる: `R_joint` = 右肘、`R_upperarm` = 肩から肘へ 2 割、`R_forearm` = 肘から手首へ 9 割、
   `chest` = 両肩の中点（下半身は腰・膝・足首で同様）。MediaPipe の点は関節の中心（腕の中）だが、マーカーは腕の
   **外側の表面**に付いているので、手足の点は胴体から離れる向きに「肩幅 × ずらし量」だけずらしてから対応させる。
   ずらし量は記録ごとに誤差が最小になるものを選ぶ（`overlay.py` の `OUTWARDS`）
3. 同じ時刻の「マーカーの 3D 座標」と「動画上の位置」の組から、カメラの位置と向きを `solvePnP` で求める。
   焦点距離は C920 の公称値（640×480 で 500 px）に固定、カメラの遅れ（動画とモーションの時刻のずれ）は探す
4. 以後は各コマの時刻のマーカーをそのカメラで投影して描く（MediaPipe の骨格の代わりに実測の骨格を出す）

結果は `overlay_calib.json`。2026-10-27 の 30 秒の記録では、左右を直し外側へのずらし（肩幅 × 0.20 ≈ 7 cm）を入れて**誤差 4.7 px（中央値）**。ずらさないと重ねた骨格が腕の中心を通り、誤差も 6.4 px だった。

- マーカーは体の正面のほぼ 1 枚の面に並ぶので、焦点距離は記録からは決まらない（探すと上限に張り付いた）。だから公称値に固定している。
  同じ理由で「正面から」と「左右を入れ替えて背中から」も見分けにくいので、ラベルどおり（`R_*` = 本人の右）を基本にしている。
- 対応の比（2 割・9 割）は体の寸法からの目安。マーカーの付け方を変えたら `overlay.py` の `MARKER_TO_MP` を直す。
- 「MediaPipe の骨格」を入れると、較正に使った対応点を白い線で重ねて見られる（コマごとに計算するので少し重い）。
- 動きの少ない記録ではカメラの遅れがほとんど決まらない。遅れ 0 より 5 % 以上良いときだけ採る。
- 「較正し直す」で最初からやり直す（`overlay_calib.json` と `pose_mediapipe.npz` を作り直す）。

MediaPipe の姿勢モデル（`models/pose_landmarker_full.task`、約 9 MB）は初回に Google から取ってくる（git には入れない）。
MediaPipe が `opencv-contrib-python` を求めるので、OpenCV はそちらに替えた（`opencv-python` と同じ `cv2`。両方入れると壊れる）。

### Motive のラベルを左右逆に付けて撮ったとき

```bash
uv run fix_lr.py <記録フォルダ名>          # R_* と L_* の名前を入れ替える（数値はそのまま）
uv run fix_lr.py <記録フォルダ名> --undo   # 元に戻す
```

`motion.csv` と `ber_motion.csv` の列名、`motion_conditions.json` の `marker_names` だけを入れ替え、元のファイルは
`_before_fix_lr/` に残す。`motion_conditions.json` に `lr_fixed`（直した日時）が入る。動画への重ね合わせの較正
（`overlay_calib.json`）は消すので、次に開いたとき作り直される。

2026-10-27 の `…_30s` はこれで直した。直す前は較正の誤差が 8.0 px、直した後は 6.4 px で、カメラの位置も床から約 1.3 m
（三脚の高さ）になった。左右を取り違えたままでも較正は「背中側から見たカメラ」として解けてしまうので、
**誤差の大きさだけでは気づけない**。動画で赤（右）の骨格が本人の右腕に乗っているかを見て確かめる。

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
| `fix_lr.py` | ラベルを左右逆に付けて撮った記録の、マーカー名の左右を入れ替える |
| `replay_viewer.html` / `overlay.py` | 再生専用の画面（`--replay`） ／ 動画にマーカーを重ねるためのカメラの較正（MediaPipe） |
| `camera.py` | Web カメラ: 画面への MJPEG、記録フォルダへの録画（コマごとの時刻つき） |
| `phase_control.py` / `phase_model_example.py` | 位相の自動調整の差し込み口とひな形 |
| `tests/test_motion_ber.py` | 装置無しのテスト |
