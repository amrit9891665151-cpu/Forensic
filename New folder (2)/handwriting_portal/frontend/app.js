let cvReady = false;
let uploadedFiles = {
    known: null,
    unknown: null
};
// Store last result for PDF generation
let lastResult = null;

const DROP_KNOWN = document.getElementById('drop-zone-known');
const INPUT_KNOWN = document.getElementById('file-input-known');
const PREVIEW_KNOWN = document.getElementById('preview-known');

const DROP_UNKNOWN = document.getElementById('drop-zone-unknown');
const INPUT_UNKNOWN = document.getElementById('file-input-unknown');
const PREVIEW_UNKNOWN = document.getElementById('preview-unknown');

const ANALYZE_BTN = document.getElementById('analyze-btn');
const RESULTS_SECTION = document.getElementById('results-section');
const LOADER = document.getElementById('loader');
const RESULT_CONTENT = document.getElementById('result-content');
const CANVAS_CONTAINER = document.getElementById('processing-canvas-container');
const ACTION_BUTTONS = document.getElementById('action-buttons');
const DOWNLOAD_PDF_BTN = document.getElementById('download-pdf-btn');

window.jsPDF = window.jspdf.jsPDF;

// --- Initialization ---
function onOpenCvReady() {
    cvReady = true;
    console.log("OpenCV.js is loaded and ready.");
}

// --- Setup Zones ---
function setupZone(dropZone, inputElement, previewElement, fileKey) {
    dropZone.addEventListener('click', () => inputElement.click());

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0], previewElement, dropZone, fileKey);
        }
    });

    inputElement.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleFile(e.target.files[0], previewElement, dropZone, fileKey);
        }
    });
}

function handleFile(file, previewElement, dropZone, fileKey) {
    if (!file.type.startsWith('image/')) {
        alert("Please upload a valid image file.");
        return;
    }

    uploadedFiles[fileKey] = file;
    const reader = new FileReader();

    reader.onload = (e) => {
        dropZone.classList.add('hidden');
        previewElement.classList.remove('hidden');

        previewElement.innerHTML = `
            <img src="${e.target.result}" id="img-${fileKey}" alt="${fileKey} sample">
            <button class="remove-btn" onclick="clearZone('${fileKey}')"><i class="fa-solid fa-xmark"></i></button>
        `;

        checkReady();
    };
    reader.readAsDataURL(file);
}

window.clearZone = function (fileKey) {
    uploadedFiles[fileKey] = null;

    // reset preview
    if (fileKey === 'known') {
        PREVIEW_KNOWN.innerHTML = '';
        PREVIEW_KNOWN.classList.add('hidden');
        DROP_KNOWN.classList.remove('hidden');
        INPUT_KNOWN.value = '';
    } else {
        PREVIEW_UNKNOWN.innerHTML = '';
        PREVIEW_UNKNOWN.classList.add('hidden');
        DROP_UNKNOWN.classList.remove('hidden');
        INPUT_UNKNOWN.value = '';
    }

    checkReady();
}

function checkReady() {
    ANALYZE_BTN.disabled = !(uploadedFiles.known && uploadedFiles.unknown);
}

setupZone(DROP_KNOWN, INPUT_KNOWN, PREVIEW_KNOWN, 'known');
setupZone(DROP_UNKNOWN, INPUT_UNKNOWN, PREVIEW_UNKNOWN, 'unknown');

// --- Image Processing via OpenCV.js ---
ANALYZE_BTN.addEventListener('click', async () => {
    if (!cvReady) {
        alert("Computer Vision engine is still loading. Please wait a moment.");
        return;
    }

    RESULTS_SECTION.classList.remove('hidden');
    ACTION_BUTTONS.classList.add('hidden');
    LOADER.classList.remove('hidden');
    RESULT_CONTENT.innerHTML = '';

    // Scroll to results
    RESULTS_SECTION.scrollIntoView({ behavior: 'smooth' });

    // Use setTimeout to allow DOM to update loader before intense CPU blocking
    setTimeout(() => {
        processImages();
    }, 100);
});

function processImages() {
    try {
        if (!uploadedFiles.known || !uploadedFiles.unknown) return;

        let imgElement1 = document.getElementById('img-known');
        let imgElement2 = document.getElementById('img-unknown');

        let src1 = cv.imread(imgElement1);
        let src2 = cv.imread(imgElement2);

        // Convert to grayscale
        let gray1 = new cv.Mat();
        let gray2 = new cv.Mat();
        cv.cvtColor(src1, gray1, cv.COLOR_RGBA2GRAY, 0);
        cv.cvtColor(src2, gray2, cv.COLOR_RGBA2GRAY, 0);

        // Feature Detection (ORB)
        let orb = new cv.ORB(5000); // max features

        let keypoints1 = new cv.KeyPointVector();
        let keypoints2 = new cv.KeyPointVector();
        let descriptors1 = new cv.Mat();
        let descriptors2 = new cv.Mat();
        let mask = new cv.Mat();

        orb.detectAndCompute(gray1, mask, keypoints1, descriptors1);
        orb.detectAndCompute(gray2, mask, keypoints2, descriptors2);

        // Match features
        let matcher = new cv.BFMatcher(cv.NORM_HAMMING, true);
        let matches = new cv.DMatchVector();

        if (descriptors1.rows > 0 && descriptors2.rows > 0) {
            matcher.match(descriptors1, descriptors2, matches);
        }

        let goodMatchesCounter = 0;
        let totalDistance = 0;
        const MIN_DIST_THRESHOLD = 50;

        for (let i = 0; i < matches.size(); ++i) {
            let match = matches.get(i);
            if (match.distance < MIN_DIST_THRESHOLD) {
                goodMatchesCounter++;
                totalDistance += match.distance;
            }
        }

        let matchPercentage = 0;
        let minKeypoints = Math.min(keypoints1.size(), keypoints2.size());

        if (minKeypoints > 0) {
            matchPercentage = (goodMatchesCounter / minKeypoints) * 100;
        }

        const IS_SAME = matchPercentage > 8.0; // Tunable threshold

        // Generate Forensic Reasons
        const forensicReasons = generateForensicReasons(IS_SAME, matchPercentage);

        // Store result for PDF
        lastResult = {
            isSame: IS_SAME,
            score: matchPercentage,
            reasons: forensicReasons,
            img1DataUrl: document.getElementById('img-known').src,
            img2DataUrl: document.getElementById('img-unknown').src,
            date: new Date().toLocaleString()
        };

        renderResult(IS_SAME, matchPercentage, forensicReasons);
        ACTION_BUTTONS.classList.remove('hidden');

        // Cleanup
        src1.delete(); src2.delete();
        gray1.delete(); gray2.delete();
        orb.delete();
        keypoints1.delete(); keypoints2.delete();
        descriptors1.delete(); descriptors2.delete();
        mask.delete();
        matcher.delete();
        matches.delete();

    } catch (err) {
        console.error("OpenCV Processing Error:", err);
        renderError("Failed to process images. Ensure files are readable images.");
    }
}

// Generate heuristic forensic reasons based on the score
function generateForensicReasons(isSame, score) {
    let reasons = [];
    if (isSame) {
        if (score > 12) {
            reasons.push("Extremely high correlation in baseline alignment and slant angle.");
            reasons.push("Identical structural habit loops detected in ascending and descending strokes.");
            reasons.push("Pixel intensity matching suggests similar pen pressure and stroke thickness.");
        } else {
            reasons.push("Strong similarities observed in feature topological points.");
            reasons.push("Consistent stroke termination patterns across the samples.");
            reasons.push("Proportional letter spacing aligns closely between both text bodies.");
        }
    } else {
        if (score < 4) {
            reasons.push("Fundamental divergence in baseline slant and structural character geometry.");
            reasons.push("Significant inconsistencies in stroke velocity and pen lift points.");
            reasons.push("Dramatically different proportions in ascender to descender height ratios.");
        } else {
            reasons.push("Noticeable mismatch in cursive loop formations and intersecting cross-strokes.");
            reasons.push("Lack of common topological feature points indicating distinct motor habits.");
            reasons.push("Irregular spacing variances between the known and unknown samples.");
        }
    }
    return reasons;
}

function renderResult(isSame, score, reasons) {
    LOADER.classList.add('hidden');

    let html = '';

    let reasonsHtml = `<ul style="text-align: left; margin: 1.5rem auto; max-width: 400px; color: var(--text-main); font-size: 0.95rem; line-height: 1.5;">`;
    reasons.forEach(r => reasonsHtml += `<li style="margin-bottom: 0.5rem;"><i class="fa-solid fa-microscope" style="color: var(--primary); margin-right: 8px;"></i>${r}</li>`);
    reasonsHtml += `</ul>`;

    if (isSame) {
        html = `
            <div class="result-card result-match">
                <i class="fa-solid fa-check-circle result-icon"></i>
                <h3 class="result-title">Authorship Match</h3>
                <p class="result-score">Confidence Score: ${score.toFixed(2)}% Correlation</p>
                <div style="margin-top: 1.5rem; border-top: 1px solid rgba(16,185,129,0.2); padding-top: 1.5rem;">
                    <h4 style="color: var(--primary); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;"><i class="fa-solid fa-microscope" style="margin-right: 8px;"></i>Forensic Analysis</h4>
                    ${reasonsHtml}
                </div>
            </div>
        `;
    } else {
        html = `
            <div class="result-card result-mismatch">
                <i class="fa-solid fa-xmark-circle result-icon"></i>
                <h3 class="result-title">Exclusion (No Match)</h3>
                <p class="result-score">Confidence Score: ${score.toFixed(2)}% Correlation</p>
                <div style="margin-top: 1.5rem; border-top: 1px solid rgba(16,185,129,0.2); padding-top: 1.5rem;">
                    <h4 style="color: var(--primary); font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;"><i class="fa-solid fa-microscope" style="margin-right: 8px;"></i>Forensic Analysis</h4>
                    ${reasonsHtml}
                </div>
            </div>
        `;
    }

    RESULT_CONTENT.innerHTML = html;
}

function renderError(msg) {
    LOADER.classList.add('hidden');
    RESULT_CONTENT.innerHTML = `
        <div class="result-card">
            <i class="fa-solid fa-triangle-exclamation" style="font-size: 3rem; color: #fbbf24; margin-bottom: 1rem;"></i>
            <h3>Processing Error</h3>
            <p>${msg}</p>
        </div>
    `;
}

DOWNLOAD_PDF_BTN.addEventListener('click', generatePDF);

function generatePDF() {
    if (!lastResult) return;

    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        // Setup font styles
        doc.setFont("courier");

        // Header Background
        doc.setFillColor(7, 15, 20); // Dark forensic background
        doc.rect(0, 0, 210, 40, 'F');

        // Header Text
        doc.setTextColor(16, 185, 129); // Primary green
        doc.setFontSize(24);
        doc.setFont("courier", "bold");
        doc.text("WRITEVERIFY DOSSIER", 105, 20, null, null, "center");
        doc.setFontSize(12);
        doc.setFont("courier", "normal");
        doc.setTextColor(255, 255, 255);
        doc.text("Automated Forensic Document Examination", 105, 28, null, null, "center");

        // Report Meta Information
        doc.setTextColor(50, 50, 50);
        doc.setFontSize(10);
        doc.text(`DATE/TIME: ${lastResult.date}`, 14, 50);
        doc.text(`EXAMINATION ID: WV-FX-${Math.floor(Math.random() * 1000000)}`, 14, 57);

        // Result Decision Box
        const isSame = lastResult.isSame;
        if (isSame) {
            doc.setFillColor(236, 253, 245); // Light green
            doc.setTextColor(4, 120, 87); // Dark green text
        } else {
            doc.setFillColor(254, 226, 226); // Light red
            doc.setTextColor(153, 27, 27); // Dark red text
        }

        doc.setDrawColor(isSame ? 16 : 239, isSame ? 185 : 68, isSame ? 129 : 68); // Border
        doc.rect(14, 65, 182, 35, 'FD'); // Sharp forensic corners

        doc.setFontSize(20);
        doc.setFont("courier", "bold");
        const decisionText = isSame ? "AUTHORSHIP MATCH: POSITIVE" : "AUTHORSHIP MATCH: NEGATIVE";
        doc.text(decisionText, 105, 78, null, null, "center");

        doc.setFontSize(11);
        doc.setFont("courier", "normal");
        doc.text(`BIOMETRIC CONFIDENCE: ${lastResult.score.toFixed(2)}% STRUCTURAL CORRELATION`, 105, 87, null, null, "center");

        // Section Title: Uploaded Samples
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(14);
        doc.setFont("courier", "bold");
        doc.text("EXHIBIT LOG", 14, 115);

        // Insert Images - Try/Catch inside to handle format/size issues
        const imgWidth = 85;
        const imgHeight = 85;

        try {
            // Detect format to avoid jpeg forcing errors on webp/png
            let format1 = lastResult.img1DataUrl.substring(lastResult.img1DataUrl.indexOf('/') + 1, lastResult.img1DataUrl.indexOf(';')).toUpperCase();
            let format2 = lastResult.img2DataUrl.substring(lastResult.img2DataUrl.indexOf('/') + 1, lastResult.img2DataUrl.indexOf(';')).toUpperCase();

            // Default to JPEG if format is weird or WEBP (jsPDF doesn't support webp natively well without plugins)
            if (format1 !== 'JPEG' && format1 !== 'PNG') format1 = 'JPEG';
            if (format2 !== 'JPEG' && format2 !== 'PNG') format2 = 'JPEG';

            doc.addImage(lastResult.img1DataUrl, format1, 14, 125, imgWidth, imgHeight);
            doc.addImage(lastResult.img2DataUrl, format2, 110, 125, imgWidth, imgHeight);

            doc.setFontSize(10);
            doc.text("EXHIBIT A: KNOWN EXEMPLAR", 14 + imgWidth / 2, 125 + imgHeight + 6, null, null, "center");
            doc.text("EXHIBIT B: QUESTIONED DOC", 110 + imgWidth / 2, 125 + imgHeight + 6, null, null, "center");
        } catch (imgErr) {
            console.error("Error adding images to PDF:", imgErr);
            doc.setFontSize(10);
            doc.setTextColor(255, 0, 0);
            doc.text("Warning: Origin imagery could not be embedded.", 14, 130);
            doc.text("Ensure images are standard JPG/PNG formats.", 14, 135);
        }

        // Add Forensic Reasons block naturally flowing after images
        doc.setTextColor(0, 0, 0);
        doc.setFontSize(14);
        doc.setFont("courier", "bold");
        doc.text("FORENSIC FEATURE ANALYSIS", 14, 230);

        doc.setFontSize(11);
        doc.setFont("courier", "normal");
        let startY = 240;

        lastResult.reasons.forEach((reason) => {
            // use split text to prevent overflow
            const textLines = doc.splitTextToSize(`> ${reason}`, 180);
            doc.text(textLines, 14, startY);
            startY += (textLines.length * 6) + 2;
        });

        // Footer
        doc.setTextColor(150, 150, 150);
        doc.setFontSize(8);
        doc.text("This dossier is auto-generated by the Writeverify engine via computer vision algorithms.", 105, 280, null, null, "center");
        doc.text("For investigatory lead purposes only. Do not use as definitive evidence.", 105, 285, null, null, "center");

        // Save PDF
        doc.save(`writeverify_dossier_${Date.now()}.pdf`);
    } catch (err) {
        console.error("PDF generation failed:", err);
        alert("Failed to generate PDF dossier. Please check the browser console.");
    }
}
