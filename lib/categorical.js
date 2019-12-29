const tf = require ('@tensorflow/tfjs-core');

class Categorical {
    constructor (B) {
        this._states = B.shape[0];
        this._tokens = B.shape[1];

        this._B = B;
    }

    pdf (data) {
        const pdf = B.gather (data.asType('int32'), -1);
        return pdf.transpose ();
    }

    sample (states, { seed }) {
        const sample_list = [];
        const states_list = tf.unstack (states);

        for (let n = 0; n < states.shape[0]; n++) {
            const state = states_list[n];
            const probs = B.gather ([state]);

            const sample = tf.multinomial (probs, 1, seed);
            sample_list.push (sample);
        }

        return tf.stack (sample_list).flatten ();
    }
}

module.exports = Categorical;
