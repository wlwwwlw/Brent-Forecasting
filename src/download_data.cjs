const { getHistoricalRates } = require('dukascopy-node');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

const root = path.resolve(__dirname, '..');
const output = path.join(root, 'data', 'recheck');
const digest = data => crypto.createHash('sha256').update(data).digest('hex');

async function main() {
  if (process.argv.includes('--full-h1')) {
    const raw = path.join(root, 'data', 'raw');
    fs.mkdirSync(raw, { recursive: true });
    const config = {instrument:'brentcmdusd',dates:{from:'2025-10-28',to:'2026-10-28'},
      timeframe:'h1',priceType:'bid',format:'json',ignoreFlats:false,volumes:true,
      utcOffset:0,retryCount:3,failAfterRetryCount:true,batchSize:10,pauseBetweenBatchesMs:1000};
    const candles = await getHistoricalRates(config);
    if (!candles.length) throw Error('Empty hourly archive');
    const payload = JSON.stringify(candles);
    const file = path.join(raw, 'brent_h1_bid_2025-10-28_2026-10-28.json');
    if (fs.existsSync(file) && digest(fs.readFileSync(file)) !== digest(payload)) {
      throw Error('Hourly source changed; existing snapshot retained');
    }
    fs.writeFileSync(file, payload);
    fs.writeFileSync(path.join(raw, 'h1_metadata.json'), JSON.stringify({
      retrieved_at_utc:new Date().toISOString(),downloader:'dukascopy-node@1.51.1',
      config,rows:candles.length,sha256:digest(payload),
      timestamp:'UTC start of each hourly candle; end is timestamp + 1 hour'
    }, null, 2));
    console.log(`Downloaded complete H1 archive: ${candles.length} rows`);
    return;
  }
  fs.mkdirSync(output, { recursive: true });
  const config = {instrument:'brentcmdusd',dates:{from:'2025-10-28',to:'2026-10-28'},
    timeframe:'d1',priceType:'bid',format:'json',ignoreFlats:false,volumes:true,
    retryCount:2,failAfterRetryCount:true};
  const rows = await getHistoricalRates(config);
  if (rows.length !== 365) throw Error(`Expected 365 calendar rows, got ${rows.length}`);
  const csv = 'timestamp_utc,open,high,low,close,volume\n' + rows.map(r =>
    [new Date(r.timestamp).toISOString(),r.open,r.high,r.low,r.close,r.volume].join(',')).join('\n')+'\n';
  fs.writeFileSync(path.join(output,'daily.csv'),csv);
  const original=fs.readFileSync(path.join(root,'data/raw/brent_d1_bid_2025-10-28_2026-10-28.csv'));
  const report={retrieved_at_utc:new Date().toISOString(),downloader:'dukascopy-node@1.51.1',
    config,rows:rows.length,sha256:digest(csv),original_sha256:digest(original),byte_identical:digest(csv)===digest(original)};
  fs.writeFileSync(path.join(output,'download_check.json'),JSON.stringify(report,null,2));
  if (!report.byte_identical) throw Error('Source differs: inspect data/recheck; original snapshot retained');
  console.log(`Daily re-download identical: ${report.sha256}`);
  const active=rows.filter(r=>r.volume>0);
  const changes=active.slice(1).map((r,i)=>({date:new Date(r.timestamp).toISOString().slice(0,10),
    magnitude:Math.abs(r.close/active[i].close-1)})).sort((a,b)=>b.magnitude-a.magnitude).slice(0,10);
  const dates=[...new Set([...changes.map(r=>r.date),'2025-12-25','2026-01-01','2026-04-03','2026-10-27'])].sort();
  const hourly=[];
  for (const date of dates) {
    const next=new Date(`${date}T00:00:00Z`);next.setUTCDate(next.getUTCDate()+1);
    const candles=await getHistoricalRates({...config,dates:{from:date,to:next.toISOString()},timeframe:'h1'});
    hourly.push({date,candles});
    console.log(`Verified download ${date}: ${candles.length} hourly rows`);
  }
  fs.writeFileSync(path.join(output,'anomaly_hourly.json'),JSON.stringify(hourly));
}
main().catch(error=>{console.error(error);process.exitCode=1;});
