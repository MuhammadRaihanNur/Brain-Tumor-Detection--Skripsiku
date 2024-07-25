let abortController = null;

async function useData() {
    const form = document.getElementById('dataForm');
    const formData = new FormData(form);

    const data = {};
    formData.forEach((value, key) => {
        data[key] = parseFloat(value);  // Convert input values to floats
    });

    abortController = new AbortController();

    try {
        // Show the loading modal
        $('#loadingModal').modal('show');

        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data),
            signal: abortController.signal
        });

        const result = await response.json();

        // Update modal content with prediction results
        $('#HasilPredictionDT').text(`Hasil Prediksi Model DT: ${result.Hasil_Prediksi_DT}`);
        $('#HasilPredictionNB').text(`Hasil Prediksi Model NB: ${result.Hasil_Prediksi_NB}`);
        $('#HasilPrediction').text(`Hasil Akhir Putusan Model: ${result.Hasil_Akhir}`);
        $('#ModelDigunakan').text(`Model Digunakan Adalah: ${result.Model_Yang}`);

        // Delay 3 seconds before showing the prediction modal
        setTimeout(() => {
            // Show the prediction modal
            $('#predictionModal').modal('show');

            // Hide the loading modal
            $('#loadingModal').modal('hide');
        }, 3000);
    } catch (error) {
        console.error('Error during prediction:', error);
        alert('Terjadi kesalahan saat memprediksi, silakan coba lagi.');
    } finally {
        // Hide the loading modal
        $('#loadingModal').modal('hide');
    }
}

function cancelOperation() {
    // Abort the fetch request if it's still ongoing
    if (abortController) {
        abortController.abort();
        abortController = null;
    }

    // Show cancel modal
    $('#cancelModal').modal('show');
}
