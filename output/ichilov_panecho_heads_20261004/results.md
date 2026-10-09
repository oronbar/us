# PanEcho measurement-head prediction test

                        model   n  events      auc       ap  ppv_top20    brier
           compact_plus_heads 182      38 0.682749 0.359673   0.351351 0.156488
   compact_plus_heads_history 182      38 0.652961 0.349671   0.270270 0.156678
            compact_reference 182      38 0.659357 0.351820   0.351351 0.156502
                heads_current 182      38 0.404788 0.179522   0.162162 0.170534
         heads_current_late25 182      38 0.719481 0.381645   0.351351 0.173047
                heads_history 182      38 0.565789 0.223795   0.135135 0.168312
         heads_history_late25 182      38 0.720029 0.399960   0.324324 0.172314
retained_cnn_moment_reference 182      38 0.725512 0.379208   0.351351 0.191608

Paired changes:

                     model                     reference  delta_auc  delta_auc_lo  delta_auc_hi  delta_ap  delta_ap_lo  delta_ap_hi
        compact_plus_heads             compact_reference   0.023392     -0.024618      0.076213  0.007853    -0.065045     0.084908
compact_plus_heads_history             compact_reference  -0.006396     -0.071093      0.058010 -0.002149    -0.087351     0.089672
      heads_current_late25 retained_cnn_moment_reference  -0.006031     -0.014753      0.002833  0.002437    -0.022060     0.029104
      heads_history_late25 retained_cnn_moment_reference  -0.005482     -0.020782      0.007394  0.020751    -0.020443     0.047540
