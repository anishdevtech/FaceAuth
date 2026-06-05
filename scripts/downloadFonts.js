const fs = require('fs');
const https = require('https');
const path = require('path');

const fonts = {
  'BarlowCondensed_600SemiBold.ttf': 'https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-SemiBold.ttf',
  'BarlowCondensed_700Bold.ttf': 'https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-Bold.ttf',
  'BarlowCondensed_800ExtraBold.ttf': 'https://github.com/google/fonts/raw/main/ofl/barlowcondensed/BarlowCondensed-ExtraBold.ttf',
  'DMSans_400Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans-Regular.ttf',
  'DMSans_500Medium.ttf': 'https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans-Medium.ttf',
  'DMSans_600SemiBold.ttf': 'https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans-SemiBold.ttf',
  'DMSans_700Bold.ttf': 'https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans-Bold.ttf',
  'DMSans_800ExtraBold.ttf': 'https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans-ExtraBold.ttf',
  'IBMPlexMono_400Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-Regular.ttf',
  'IBMPlexMono_600SemiBold.ttf': 'https://github.com/google/fonts/raw/main/ofl/ibmplexmono/IBMPlexMono-SemiBold.ttf'
};

const dir = path.join(__dirname, '..', 'assets', 'fonts');

if (!fs.existsSync(dir)){
    fs.mkdirSync(dir, { recursive: true });
}

Object.entries(fonts).forEach(([filename, url]) => {
  const filePath = path.join(dir, filename);
  https.get(url, (res) => {
    if (res.statusCode === 302 || res.statusCode === 301) {
       https.get(res.headers.location, (redirectRes) => {
           const file = fs.createWriteStream(filePath);
           redirectRes.pipe(file);
           file.on('finish', () => { file.close(); console.log(`Downloaded ${filename}`); });
       });
    } else {
       const file = fs.createWriteStream(filePath);
       res.pipe(file);
       file.on('finish', () => { file.close(); console.log(`Downloaded ${filename}`); });
    }
  }).on('error', (err) => {
    console.error(`Error downloading ${filename}: ${err.message}`);
  });
});
