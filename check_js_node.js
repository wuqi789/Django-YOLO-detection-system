const fs = require('fs');
const content = fs.readFileSync('d:/webtest/Django-YOLO-detection-system-main/detection/templates/index.html', 'utf8');

// Find the first script block
const scriptStart = content.indexOf('<script>');
const scriptEnd = content.indexOf('</script>', scriptStart);
const scriptContent = content.substring(scriptStart + 8, scriptEnd);

try {
    new Function(scriptContent);
    console.log('No syntax error');
} catch(e) {
    console.log('Error:', e.message);
    
    // Try to find the position by binary search
    let low = 0;
    let high = scriptContent.length;
    let errorPos = high;
    
    while (low < high) {
        const mid = Math.floor((low + high) / 2);
        try {
            new Function(scriptContent.substring(0, mid));
            low = mid + 1;
        } catch(err) {
            errorPos = mid;
            high = mid;
        }
    }
    
    // Find line number
    const beforeError = scriptContent.substring(0, errorPos);
    const lineNum = beforeError.split('\n').length;
    const lines = scriptContent.split('\n');
    
    console.log('Error around line', lineNum, 'in script');
    console.log('Context:');
    for (let i = Math.max(0, lineNum - 3); i < Math.min(lines.length, lineNum + 3); i++) {
        console.log((i + 1) + ':', lines[i].substring(0, 100));
    }
}
