/** Export audited, static research summary tables in the requested CSV format. */
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const [payloadPath, outputDir] = process.argv.slice(2);
if (!payloadPath || !outputDir || !process.env.ARTIFACT_NODE_MODULES) {
  throw new Error('Usage: ARTIFACT_NODE_MODULES=... node export_tables.mjs payload.json output_directory');
}
const resolveRuntime = createRequire(path.join(process.env.ARTIFACT_NODE_MODULES, '..', 'artifact-resolver.cjs'));
const { Workbook } = await import(pathToFileURL(resolveRuntime.resolve('@oai/artifact-tool')).href);
const payload = JSON.parse(await fs.readFile(payloadPath, 'utf8'));
const workbook = Workbook.create();
const exports = [];
const frontColumns = ['rank_F1','rank_F1_mean','model','split','context','threshold_panel','threshold_mode','threshold',
  'F1','precision','recall','AP','ROC_AUC','n_total','TP','FP','FN','TN'];
const quote = (v) => {
  const text = v == null ? '' : String(v);
  return /[",\r\n]/.test(text) ? '"' + text.replaceAll('"', '""') + '"' : text;
};
let index = 0;
for (const [filename, rows] of Object.entries(payload.tables)) {
  const allColumns = [...new Set(rows.flatMap(r => Object.keys(r)))];
  const columns = [...frontColumns.filter(c => allColumns.includes(c)),
    ...allColumns.filter(c => !frontColumns.includes(c))];
  const matrix = [columns, ...rows.map(r => columns.map(c => r[c] ?? null))];
  const sheet = workbook.worksheets.add('Table' + (++index));
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, columns.length);
  range.values = matrix;
  exports.push({filename, range, expectedRows: rows.length, columns: columns.length});
}
workbook.recalculate();
const manifest = [];
for (const entry of exports) {
  const matrix = entry.range.values;
  if (matrix.length !== entry.expectedRows + 1) throw new Error('Row count changed: ' + entry.filename);
  const content = matrix.map(row => row.map(quote).join(',')).join('\r\n') + '\r\n';
  await fs.writeFile(path.join(outputDir, entry.filename), '\ufeff' + content, {encoding:'utf8', flag:'wx'});
  manifest.push({file:entry.filename, rows:entry.expectedRows, columns:entry.columns,
    sha256:createHash('sha256').update('\ufeff'+content,'utf8').digest('hex')});
}
await fs.writeFile(path.join(outputDir, 'SUMMARY_CONFIG.json'), JSON.stringify(payload.config,null,2), {flag:'wx'});
await fs.writeFile(path.join(outputDir, 'OUTPUT_MANIFEST.json'), JSON.stringify(manifest,null,2), {flag:'wx'});
console.log(JSON.stringify({exported:manifest, no_model_training:true}));
