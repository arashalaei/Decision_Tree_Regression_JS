/* jslint esversion: 9 */

/**
 * @author Arash Alaei <arashalaei22@gmail.com>
 * @since 02/00/2021
 */

require('@tensorflow/tfjs-node');
const dfd = require('danfojs-node'); // import pandas as pd
const DTRegression  = require('ml-cart').DecisionTreeRegression;
const tf = require('@tensorflow/tfjs'); // import numpy as np
const path = require("path");

(async() => {
    // Importing the dataset
    const dataset = await dfd.read_csv(`file://${path.join(__dirname,'./Position_Salaries.csv')}`);

    let x = (dataset.iloc({rows:[':'], columns:['1']}).values);
    let y = tf.util.flatten(dataset.iloc({rows:[':'], columns:['2']}).values);
    console.log(x);
    console.log(y);
    const reg = new DTRegression();
    reg.train(x, y);
    console.log(reg.predict([[6.5]]));
})();

