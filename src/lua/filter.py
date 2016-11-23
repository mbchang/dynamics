for line in sorted(open('blstmtower.txt','r').readlines()):
    if 'mixed_n6_t60_ex50000_m_z_o_dras3_rda__mixed_n6_t60_ex50000_m_z_o_dras3' in line and 'modelbl' not in line:
        print line.strip()