const assert = require('node:assert/strict');
const Library = require('../apps/web/static/library.js');
const catalog = {garments: {pants: {slot: 'lower', label: 'Pants'}, shirt: {slot: 'upper', label: 'Shirt'}}, motions: {walk: {label: 'Walk'}}};
const common = {garments: ['pants', 'shirt'], gender: 'male', height: 175, weight: 75, motion: 'walk'};
const lib = Library.build({videos: [
  {...common, path: 'opaque.mp4'},
  {...common, path: 'upper.mp4', opacity: .5, opacity_mode: 'v3', opacity_garments: ['shirt']},
  {...common, path: 'all.mp4', opacity: .5, opacity_mode: 'v3'},
  {...common, path: 'direct.mp4', opacity: .5, opacity_mode: 'direct'},
  {...common, path: '../secret.mp4'},
]}, catalog, '/library/media/');
const sel = {gender: 'male', height: '175', weight: '75', upper: 'shirt', lower: 'pants', motion: 'walk'};
assert.equal(lib.videoUrl(sel), '/library/media/opaque.mp4');
assert.equal(lib.videoUrl({...sel, opacity: '0.5:shirt'}), '/library/media/upper.mp4');
assert.equal(lib.videoUrl({...sel, opacity: '0.5:'}), '/library/media/all.mp4');
assert.equal(lib.videoUrl({...sel, weight: '80'}), null);
assert.equal(lib.videos.length, 3);
assert.equal(lib.opacities(sel).length, 3);
assert.equal(lib.labelOf('opacity', '0.5:shirt'), '50% Shirt');
assert.equal(lib.normalize({...sel, opacity: '0.3:shirt'}).opacity, null);
assert.equal(lib.genders(sel).find(g => g.id === 'female').disabled, true);
console.log('Video lookup, target-specific opacity, unavailable combinations and URL guards passed');
