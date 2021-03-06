#!/usr/bin/env python3

pretest = {
    'PX[PXL]':'P',
    'PX[PXR]':'silence',
    'XT[XTL]':'silence',
    'XT[XTR]':'T',
    'KX[KXL]':'K',
    'KX[KXR]':'silence',
    'XB[XBL]':'silence',
    'XB[XBR]':'B',
    'DX[DXL]':'D',
    'DX[DXR]':'silence',
    'XG[XGL]':'silence',
    'XG[XGR]':'G'
    }

prompt_groups = (
    'PP',
    'PT',
    'PK',
    'PB',
    'PD',
    'PG',
    'TP',
    'TT',
    'TK',
    'TB',
    'TD',
    'TG',
    'KP',
    'KT',
    'KK',
    'KB',
    'KD',
    'KG',
    'BP',
    'BT',
    'BK',
    'BB',
    'BD',
    'BG',
    'DP',
    'DT',
    'DK',
    'DB',
    'DD',
    'DG',
    'GP',
    'GT',
    'GK',
    'GB',
    'GD',
    'GG'
    )

key = {
    'PP[PPL]':'P',
    'PP[PPR]':'P',
    'PT[PTL]':'P',
    'PT[PTR]':'T',
    'PK[PKL]':'P',
    'PK[PKR]':'K',
    'PB[PBL]':'P',
    'PB[PBR]':'B',
    'PD[PDL]':'P',
    'PD[PDR]':'D',
    'PG[PGL]':'P',
    'PG[PGR]':'G',
    'TP[TPL]':'T',
    'TP[TPR]':'P',
    'TT[TTL]':'T',
    'TT[TTR]':'T',
    'TK[TKL]':'T',
    'TK[TKR]':'K',
    'TB[TBL]':'T',
    'TB[TBR]':'B',
    'TD[TDL]':'T',
    'TD[TDR]':'D',
    'TG[TGL]':'T',
    'TG[TGR]':'G',
    'KP[KPL]':'K',
    'KP[KPR]':'P',
    'KT[KTL]':'K',
    'KT[KTR]':'T',
    'KK[KKL]':'K',
    'KK[KKR]':'K',
    'KB[KBL]':'K',
    'KB[KBR]':'B',
    'KD[KDL]':'K',
    'KD[KDR]':'D',
    'KG[KGL]':'K',
    'KG[KGR]':'G',
    'BP[BPL]':'B',
    'BP[BPR]':'P',
    'BT[BTL]':'B',
    'BT[BTR]':'T',
    'BK[BKL]':'B',
    'BK[BKR]':'K',
    'BB[BBL]':'B',
    'BB[BBR]':'B',
    'BD[BDL]':'B',
    'BD[BDR]':'D',
    'BG[BGL]':'B',
    'BG[BGR]':'G',
    'DP[DPL]':'D',
    'DP[DPR]':'P',
    'DT[DTL]':'D',
    'DT[DTR]':'T',
    'DK[DKL]':'D',
    'DK[DKR]':'K',
    'DB[DBL]':'D',
    'DB[DBR]':'B',
    'DD[DDL]':'D',
    'DD[DDR]':'D',
    'DG[DGL]':'D',
    'DG[DGR]':'G',
    'GP[GPL]':'G',
    'GP[GPR]':'P',
    'GT[GTL]':'G',
    'GT[GTR]':'T',
    'GK[GKL]':'G',
    'GK[GKR]':'K',
    'GB[GBL]':'G',
    'GB[GBR]':'B',
    'GD[GDL]':'G',
    'GD[GDR]':'D',
    'GG[GGL]':'G',
    'GG[GGR]':'G'
    }

def verify_answer(q_code, candidate):

    if q_code in key:
        if candidate.lower() == key[q_code].lower():
            return True
        # count empty responses as correct if silence played
        elif candidate == '' and key[q_code].lower() == 'silence':
            return True
        else:
            return False
    if q_code in pretest:
        if candidate.lower() == pretest[q_code].lower():
            return True
        elif candidate == '' and pretest[q_code].lower() == 'silence':
            return True
        else:
            return False
    else:
        return False
