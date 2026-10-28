// CSVの重複整理はPythonで行い、このファイルはExcelの表示だけを担当する。
import fs from 'node:fs/promises';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const [input, output, previewDir] = process.argv.slice(2);
const data = JSON.parse(await fs.readFile(input, 'utf8'));
const workbook = Workbook.create();
const dateFormat = 'yyyy-mm-dd hh:mm:ss';
const numericHeaders = ['総トン数', '全長', '離岸時喫水(船首)', '離岸時喫水(船尾)'];

// Excelにはタイムゾーンがないため、日本時間の年月日時分秒をそのまま表示する。
const excelDate = (iso) => new Date(`${iso.slice(0, 19)}Z`);

function writeTable(name, headers, rows, tableName, freezeColumns = 0) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const count = rows.length + 1;
  const area = sheet.getRangeByIndexes(0, 0, count, headers.length);
  // 文字列と数値を区別したまま書き込む。識別子の先頭の0も保持する。
  area.values = [headers, ...rows];
  area.format.font = { name: 'Arial', size: 10, color: '#243746' };
  area.format.rowHeight = 22;
  area.format.columnWidth = 16;
  area.format.verticalAlignment = 'center';
  area.format.wrapText = false;
  let columnNumber = headers.length;
  let lastColumn = '';
  while (columnNumber > 0) {
    columnNumber--;
    lastColumn = String.fromCharCode(65 + columnNumber % 26) + lastColumn;
    columnNumber = Math.floor(columnNumber / 26);
  }
  const table = sheet.tables.add(`A1:${lastColumn}${count}`, true, tableName);
  table.showFilterButton = true;
  const heading = sheet.getRangeByIndexes(0, 0, 1, headers.length);
  heading.format = {
    fill: '#315F7D', font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true, rowHeight: 34,
  };
  heading.format.borders = { insideVertical: { style: 'thin', color: '#FFFFFF' } };
  sheet.freezePanes.freezeRows(1);
  if (freezeColumns) sheet.freezePanes.freezeColumns(freezeColumns);
  return sheet;
}

const historyHeaders = ['CSV出力日時（JST）', '表名', '原本の行数', '同一CSV内の一意行数', '新しく蓄積した行数', '過去にもあった行数', '同一CSV内の重複行数', '元CSV', '取得元URL'];
const historyRows = data.history.map(row => [excelDate(row[0]), ...row.slice(1)]);
const history = writeTable('取得履歴', historyHeaders, historyRows, 'CollectionHistory', 2);
history.getRange(`A2:A${historyRows.length + 1}`).setNumberFormat(dateFormat);
history.getRange(`C2:G${historyRows.length + 1}`).setNumberFormat('#,##0');
history.getRange(`A1:A${historyRows.length + 1}`).format.columnWidth = 24;
history.getRange(`B1:B${historyRows.length + 1}`).format.columnWidth = 28;
history.getRange(`H1:H${historyRows.length + 1}`).format.columnWidth = 66;
history.getRange(`I1:I${historyRows.length + 1}`).format.columnWidth = 53;
history.getRange('K1').values = [['記録の読み方']];
history.getRange('K2').values = [['日時はCSVの出力名から記録。時刻は日本時間。']];
history.getRange('K3').values = [['全列が同じ行をまとめ、変更された行は別の履歴として保存。']];
history.getRange('K4').values = [['表の件数は船の隻数ではなく、予定や港内移動を含む行数。']];
history.getRange('K5').values = [['各表の入港・出港日時などは原文。年のない日時は文字列で保持。']];
history.getRange('K6').values = [['予定と実績の区別・荷役開始終了時刻は別途確認が必要。']];
history.getRange('K7').values = [['原本CSVは出力名を変えずに保存。初回・最終CSVで照合できる。']];
history.getRange('K1:K7').format.font = { name: 'Arial', size: 10, color: '#526675' };
history.getRange('K1:K7').format.columnWidth = 85;

let tableIndex = 0;
for (const [name, table] of Object.entries(data.tables)) {
  if (!table.headers.length) continue;
  // 空の見出しも列位置が分かる名前を付けて残す。記号の意味は推測しない。
  const headers = table.headers.map((value, index) => value || `原文列${index + 1}（見出しなし）`);
  const metadataHeaders = ['初回確認日時（JST）', '最終確認日時（JST）', '確認CSV数', '初回CSV', '最終CSV'];
  const rows = table.records.map(record => {
    const values = record.values.map((value, index) => {
      if (value !== '' && numericHeaders.includes(table.headers[index])) {
        const number = Number(value.replaceAll(',', ''));
        if (Number.isFinite(number)) return number;
      }
      return value;
    });
    return [...values, excelDate(record.firstSeen), excelDate(record.lastSeen), record.count, record.firstFile, record.lastFile];
  });
  const sheet = writeTable(name, [...headers, ...metadataHeaders], rows, `PortData${++tableIndex}`, name === '出港船表' ? 5 : 3);
  headers.forEach((header, index) => {
    const column = sheet.getRangeByIndexes(0, index, rows.length + 1, 1);
    if (header === '船名') column.format.columnWidth = 38;
    if (header === '代理店') column.format.columnWidth = 66;
    if (header.includes('日時') || header.includes('時刻')) column.format.columnWidth = 20;
    if (header.includes('見出しなし')) column.format.columnWidth = 20;
    if (numericHeaders.includes(header) && rows.length) {
      sheet.getRangeByIndexes(1, index, rows.length, 1).setNumberFormat(header === '総トン数' ? '#,##0' : '0.00');
    }
  });
  const start = headers.length;
  if (rows.length) sheet.getRangeByIndexes(1, start, rows.length, 2).setNumberFormat(dateFormat);
  sheet.getRangeByIndexes(0, start, rows.length + 1, 2).format.columnWidth = 24;
  sheet.getRangeByIndexes(0, start + 2, rows.length + 1, 1).format.columnWidth = 13;
  if (rows.length) sheet.getRangeByIndexes(1, start + 2, rows.length, 1).setNumberFormat('#,##0');
  sheet.getRangeByIndexes(0, start + 3, rows.length + 1, 2).format.columnWidth = 66;
}

workbook.recalculate();
console.log((await workbook.inspect({ kind: 'table', range: '取得履歴!A1:G8', include: 'values', tableMaxRows: 8, tableMaxCols: 7, maxChars: 2200 })).ndjson);
console.log((await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!', options: { useRegex: true, maxResults: 10 }, summary: 'Excelのエラー確認' })).ndjson);
if (previewDir) {
  await fs.mkdir(previewDir, { recursive: true });
  for (const name of ['取得履歴', ...Object.keys(data.tables)]) {
    if (name !== '取得履歴' && !data.tables[name].headers.length) continue;
    const preview = await workbook.render({ sheetName: name, range: name === '取得履歴' ? 'A1:G8' : 'A1:I8', scale: 1.5, format: 'png' });
    await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await preview.arrayBuffer()));
  }
}
await (await SpreadsheetFile.exportXlsx(workbook)).save(output);
