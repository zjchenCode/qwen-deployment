# Qwen2.5-VL-72B デプロイメント問題修正

## 修正された問題

### 1. メッセージ処理エラー
**問題**: `'dict' object has no attribute 'startswith'`
- **原因**: quick_deploy.shで生成されるサーバーコードに`process_content`関数が不足していた
- **修正**: 適切なメッセージ内容処理ロジックを追加

### 2. Deprecation警告
**問題**: `You have video processor config saved in preprocessor.json file which is deprecated`
- **原因**: video processor設定が古い形式で保存されている
- **修正**: 自動的に`video_preprocessor.json`ファイルを作成

### 3. GPU メモリ管理問題
**問題**: 推理後にGPUメモリが蓄積し、長時間実行後にOOMエラーが発生
- **原因**: 推理で生成されるテンソルが適切に解放されない
- **修正**: 自動メモリクリーンアップシステムを実装

### 4. タイムアウト問題
**問題**: APIリクエストが長時間実行されて524エラーが発生
- **原因**: サーバーのタイムアウト設定が不十分
- **修正**: 10分（600秒）のリクエストタイムアウトを設定

## デプロイメント方法

### 新規デプロイ
```bash
./quick_deploy.sh
```

### 既存サーバーの再起動（修正適用）
```bash
cd /workspace
python3 server.py
```

## テスト方法
```bash
./test_api.sh
```

## メモリ管理機能

### 自動メモリクリーンアップ
- **設定**: `AUTO_CLEANUP_MEMORY=true` (デフォルト有効)
- **間隔**: `CLEANUP_INTERVAL=3` (3回の推理毎)
- **モード**: `AGGRESSIVE_CLEANUP=true` (積極的クリーンアップ)
- **タイムアウト**: `REQUEST_TIMEOUT=600` (10分)

### 手動メモリクリーンアップ
```bash
curl -X POST http://localhost:8000/v1/memory/cleanup \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### メモリ監視
```bash
# ヘルスチェック（メモリ情報含む）
curl http://localhost:8000/health

# 詳細メトリクス
curl http://localhost:8000/metrics \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### 環境変数設定
```bash
export AUTO_CLEANUP_MEMORY="true"     # 自動クリーンアップ有効/無効
export CLEANUP_INTERVAL="3"           # クリーンアップ間隔（推理回数）
export AGGRESSIVE_CLEANUP="true"      # 積極的クリーンアップ（true/false）
export FORCE_SYNC="true"              # GPU同期（true/false）
export MAX_MEMORY_THRESHOLD="0.85"    # メモリ使用率閾値（0.0-1.0）
export REQUEST_TIMEOUT="600"          # リクエストタイムアウト（秒）
```

## 変更内容

### server.py
- ✅ `process_content`関数でメッセージ処理を修正
- ✅ deprecation警告の自動修正機能を追加
- ✅ 日本語コメントに更新
- ✅ **新規**: temperature 警告の修正
- ✅ **新規**: BitsAndBytes データ型警告の修正
- ✅ **新規**: transformers詳細ログ抑制設定

### quick_deploy.sh
- ✅ 生成されるサーバーコードに`process_content`関数を追加
- ✅ base64画像処理サポートを追加
- ✅ deprecation警告修正機能を追加
- ✅ 日本語メッセージに更新
- ✅ **新規**: temperature 警告の修正
- ✅ **新規**: BitsAndBytes データ型警告の修正
- ✅ **新規**: transformers詳細ログ抑制設定

### 修正された問題
1. **メッセージ処理エラー**: `process_content`関数の追加
2. **Temperature 警告**: 生成パラメータを条件付きで設定
3. **BitsAndBytes 警告**: 量子化計算型をfloat16に統一
4. **Transformers 詳細ログ**: 環境変数で抑制
5. **GPU メモリ管理**: 自動クリーンアップシステムの実装
6. **タイムアウト設定**: 10分のリクエストタイムアウトを追加

### GPU メモリ動作の説明
- **正常**: モデル重み（36-72GB）は常駐、推理キャッシュも一部保持
- **改善**: 推理毎に中間テンソルを自動削除、定期的なメモリクリーンアップ
- **監視**: リアルタイムメモリ使用量とクリーンアップ統計の提供

これで`python3 server.py`または`./quick_deploy.sh`を実行すると、警告なく、メモリ効率的に動作するはずです。 