# Step 0 — baseline reproduction and spectral-moment audit

## 1. 実験目的

直接APE回帰と低次元スペクトルモーメント経由予測を比較する前に、既存F+TO3モデル、スペクトル由来のS0/S1、APEの算式、共通評価母集団を監査する。

## 2. 設計理念

既存baselineのコード・5大気変数・seed・split・MLP・再帰推論・校正を一切変更しない。新しい処理は独立した監査スクリプトに限定し、比較対象の時刻と表面を先に固定する。

## 3. 仮説

元スペクトルに対して既存の1 nm単純和を適用すれば、pickleの瞬時APEおよびbaselineの時間算術平均APEを数値精度で再現できる。台形積分は端点重みが異なるため、単純和との差が残る可能性がある。

## 4. 使用データ

- 筑波地点301、HSR水平面とTSR傾斜32°。各日付の`10HSR*.csv`/`10TSR*.csv`から、pickleの`source_file`と`source_row_index`で元行を厳密に参照した。
- 元CSVはShift-JIS、先頭5列が地点番号・地点名・年月日・時分・リマーク、後続1351列が350–1700 nmの1 nm刻み、列単位は`W/m2/μm`。
- 解析対象は350–1050 nmの701列。既存target生成はRemark 1/2、全対象波長の有限値かつ非`-9999`、正の分子・分母、APE 1.2–2.2 eVを採用する。実際に採用されたRemark内訳はHSR {1: 76742}、TSR {1: 76958}。
- HSR/TSRともpickleの時刻はJSTのtimezone-naive値であり、監査では元CSV時刻、行番号、地点、Remarkとの一致を全採用行で検証した。入力ハッシュは`audit_manifest.json`に保存。
- 検証した元ファイルはHSR 1,633件・TSR 1,618件、採用スペクトルはHSR 76,742件・TSR 76,958件。元スペクトルは既存baseline CLIのAPE入力に含まれないため、独立監査CLIの`--hsr-spectral-root`/`--tsr-spectral-root`で明示的に受け取った。
- 添付の参照`model.py`は0バイトだったため、既存実験スクリプトと元APE生成実装を基準にした。

## 5. APE / S0 / S1の定義

`E(λ)`の単位を`W m^-2 μm^-1`、波長をnmとし、`S0 = ∫ E(λ)dλ/1000` (`W m^-2`)、`S1 = ∫ λE(λ)dλ/1000` (`W m^-2 nm`) とする。`APE = 1239.84193 S0/S1` eV。積分は`numpy.trapezoid`（利用可能なNumPy）を原則とし、既存生成式の`sum(E)`/`sum(λE)`も別列に保存した。定数`1239.84193 eV nm`は既存TSR生成実装と一致する。

## 6. hourly aggregation上の注意

- A: 各瞬時スペクトルからAPEを計算し、JSTの`[hh:00, hh+1:00)`で表面別に算術平均。既存baseline targetの再現はAで行う。
- B: 同じ時間窓のS0/S1をそれぞれ平均した後に比をとる。これはAと一般に一致しない。
- 同じ積分規則の中で比較したB−Aを以下に示す（差の単位eV）。既存の単純和規則では、HSRのMAEが2.604 meV、最大差が67.233 meV、TSRのMAEが2.772 meV、最大差が65.622 meVだった。これは単なる積分規則の差ではない。

| surface | method | n | mae_eV | max_abs_difference_eV | mean_difference_eV |
|---|---|---|---|---|---|
| HSR | trapezoid_B_minus_A | 12181 | 0.0025969846 | 0.067066892 | -0.002436552 |
| HSR | legacy_B_minus_A | 12181 | 0.0026042723 | 0.067233133 | -0.0024433947 |
| TSR | trapezoid_B_minus_A | 12181 | 0.0027642371 | 0.065470751 | -0.0024488614 |
| TSR | legacy_B_minus_A | 12181 | 0.0027716575 | 0.065621594 | -0.0024556486 |

## 7. 現行baselineモデル

Fixed-beta Ridge → Train OOF beta prediction → residual target → residual MLP → 表面別の因果的recursive residual rollout → validation後半だけによるsurface-wise affine calibration → untouched chronological Test。一次大気入力はAM、AOD550、omega(TQV)、TOTANGSTR、TO3のみ。対照HGBも既存スクリプトのまま実行した。

## 8. chronological split

| role | start_utc | end_utc | paired_hours |
|---|---|---|---|
| train_oof | 2011-06-01T00:30:00+00:00 | 2014-08-10T06:30:00+00:00 | 8526 |
| validation_early | 2014-08-11T03:30:00+00:00 | 2014-12-25T03:30:00+00:00 | 905 |
| validation_calibration | 2014-12-26T03:30:00+00:00 | 2015-05-03T03:30:00+00:00 | 910 |
| test | 2015-05-04T00:30:00+00:00 | 2015-12-31T05:30:00+00:00 | 1822 |

Residual MLPのbest epochは39、validation earlyの最良lossは0.29456541。early stoppingとsurface-wise affine calibrationは別のvalidation期間を使用し、Testは学習・選択・校正に使っていない（実行ログ上: `False`）。

seed=42、70/15/15%の時系列分割、6時間purge、Validation前半でearly stopping、後半で校正。HSR/TSR同時刻は同一role。元のsplitラベルを行単位予測CSVから継承し、共通母集団の作成時にも変更していない。

| split | surface | baseline_rows | moment_valid_rows | common_rows | excluded_rows |
|---|---|---|---|---|---|
| test | HSR | 1822 | 1822 | 1822 | 0 |
| test | TSR | 1822 | 1822 | 1822 | 0 |
| train_oof | HSR | 8526 | 8526 | 8526 | 0 |
| train_oof | TSR | 8526 | 8526 | 8526 | 0 |
| validation_calibration | HSR | 910 | 910 | 910 | 0 |
| validation_calibration | TSR | 910 | 910 | 910 | 0 |
| validation_early | HSR | 905 | 905 | 905 | 0 |
| validation_early | TSR | 905 | 905 | 905 | 0 |

## 9. APE再計算検証

stored APEを基準としたR²、MAE、最大絶対差、平均差を記録した。legacy_sumは既存生成式、trapezoidは台形積分である。時間Aは瞬時APEの算術平均、Bは平均S0/S1の比。

| surface | level | method | n | r2 | mae_eV | max_abs_difference_eV | mean_difference_eV |
|---|---|---|---|---|---|---|---|
| HSR | instantaneous | legacy_sum | 76742 | 1 | 1.6680601e-15 | 9.3258734e-15 | -2.1107287e-17 |
| HSR | instantaneous | trapezoid | 76742 | 0.9999842 | 0.00010354708 | 0.00058661436 | 9.493841e-05 |
| TSR | instantaneous | legacy_sum | 76958 | 1 | 1.4149364e-16 | 6.6613381e-16 | 3.2892077e-19 |
| TSR | instantaneous | trapezoid | 76958 | 0.99998658 | 0.00014958702 | 0.0005355343 | 0.00014529822 |
| HSR | baseline_hourly | legacy_A | 12181 | 1 | 7.1832277e-16 | 4.6629367e-15 | -2.0890166e-17 |
| HSR | baseline_hourly | legacy_B | 12181 | 0.94345849 | 0.0026042723 | 0.067233133 | -0.0024433947 |
| HSR | baseline_hourly | trapezoid_A | 12181 | 0.99998101 | 9.5293012e-05 | 0.00029170618 | 9.3773818e-05 |
| HSR | baseline_hourly | trapezoid_B | 12181 | 0.94425992 | 0.0025945572 | 0.067190572 | -0.0023427782 |
| HSR | A_vs_B | trapezoid_B_minus_A | 12181 | 0.94364433 | 0.0025969846 | 0.067066892 | -0.002436552 |
| HSR | A_vs_B | legacy_B_minus_A | 12181 | 0.94345849 | 0.0026042723 | 0.067233133 | -0.0024433947 |
| TSR | baseline_hourly | legacy_A | 12181 | 1 | 1.5837152e-16 | 1.110223e-15 | -2.6249424e-18 |
| TSR | baseline_hourly | legacy_B | 12181 | 0.96558097 | 0.0027716575 | 0.065621594 | -0.0024556486 |
| TSR | baseline_hourly | trapezoid_A | 12181 | 0.99997837 | 0.00013552644 | 0.00035676034 | 0.00013504914 |
| TSR | baseline_hourly | trapezoid_B | 12181 | 0.96621991 | 0.0027644419 | 0.065469409 | -0.0023138122 |
| TSR | A_vs_B | trapezoid_B_minus_A | 12181 | 0.96566079 | 0.0027642371 | 0.065470751 | -0.0024488614 |
| TSR | A_vs_B | legacy_B_minus_A | 12181 | 0.96558097 | 0.0027716575 | 0.065621594 | -0.0024556486 |

![APE再計算検証](ape_recalculation_validation.png)

## 10. baseline結果

READMEの既存結果と比較した独立Test。MBEは正解−予測、単位meV。

| surface | n | r2 | rmse_meV | mae_meV | mbe_truth_minus_prediction_meV | readme_consistent |
|---|---|---|---|---|---|---|
| HSR | 1822 | 0.65670173 | 13.841725 | 10.67595 | -1.006297 | True |
| TSR | 1822 | 0.77948266 | 14.190303 | 10.311606 | -0.4818648 | True |

## 11. 予測APE vs 正解APEの散布図の解釈

独立Testのみ。横軸は正解APE、縦軸は予測APE、破線は`y=x`。HSRとTSRは別パネルとした。高APE側の平均への回帰・過小予測は残るため、R²だけで上側誤差を隠さない。

![baseline独立Test散布図](baseline_measured_vs_predicted_ape.png)

## 12. common cohort

固定キー集合のSHA-256: `d48564e0ec748d7b109124b025865a4d985c5a5f3cd55c139d04e7bfd2392a0f`。時刻はUTCで保存する。

baselineに存在する4 roleのうち、元スペクトル由来のS0/S1と瞬時APEが全件一致し、同時刻HSR/TSRが両方揃う時刻だけを採用した。`common_cohort_keys.csv`を後続のBaseline/Raw Moment/Normalized Moment/Physics-guided Multi-taskの固定キー集合とする。後続モデルがこの集合以外を評価することは禁止する。baseline全role行と同一か: **True**。

## 13. 問題点・異常

- 既存APEは台形積分ではなく1 nm単純和である。両者の差は積分端点の扱いによる数値定義差であり、未解決のデータ異常と混同しない。
- BはAと異なり得る。後続のS0/S1学習でBを予測して既存AのR²と比較する設計は不公平である。
- raw prediction、スペクトル行由来のモーメント、共通キーは公開データセットではない。リポジトリの`.gitignore`で非公開とする。

## 14. 次のRaw Moment実験を実施可能か

**PASS**。legacy_sumによる瞬時/時間Aの最大差がそれぞれ9.33e-15/4.66e-15 eV。次段階では既存APEと同じ数値定義のS0/S1を教師にするか、台形積分のAPEへ全モデルのtargetを統一してbaselineを再学習する必要がある。どちらを採るか実験設定に明記する。common cohortが縮小した場合は、このcohortでbaselineを同じ条件で再学習するまで性能比較を開始しない。

## 15. 結論

既存baselineとスペクトル積分量の対応を検証した。判定: **PASS**。研究仮説に不利な台形積分差・hourly A/B差・高APE過小予測を含め、結果をそのまま報告した。
