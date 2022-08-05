import warnings

def classify(dcmObj):
    return getattr(dcmObj, 'Modality', None)


def reconstruct_series(dicom_dict):
    '''
    1. get InstanceNumber as the original indexing and make map indexing -> dicom filename
    2. check inner product of ImageOrientationPatient of first and second images: if == 0 -> discard first
    3. check alignment from ImagePositionPatient 
    4. calculate spacing
    5. retag series with ImageOrientationPatient: Axial-like, Coronal-like, Sagittal-like, Other
    6. retag contrast with ContrastBolusAgent: null -> w/o contrast, otherwise -> w/
    7. return map, dicom_dict that indexing by InstanceNumber, use_contrast, series_like, ..

    Input: dicom_dict
        dicom_dict: {<unique key(instance_no or filename)>: <FileDataset object of same SeriesDescription>}
    Return: data
        data = {
            index_list: [<index>]
            index_map: {<index>: <dicom filename>}
            index_dicom_dict: {<index>: <FileDataset>}
            series_like: <series like axial/coronal/sagittal/other>
            chirality: <True(right-handed, index increasing from front to back): ((Pn - Pn-1) dot k) > 0; vice versa>
            use_contrast: <True: w/ contrast; False: w/o contrast>
            spacing: <calculated spacing>
            ImageOrientationPatient: [<float>]
        }
    '''

    # 1. get InstanceNumber as the original indexing and make map indexing -> dicom filename
    index_map = {
            int(x.InstanceNumber): unikey
            for unikey, x in dicom_dict.items()
            if getattr(x, 'InstanceNumber', None) is not None}

    index_data_dict = {
            i: dicom_dict.get(unikey)
            for i, unikey in index_map.items()}

    index_list = list(index_map.keys())
    index_list.sort()

    mod_list = list({classify(x) for _, x in dicom_dict.items()})
    if len(mod_list) > 1:
        raise AssertionError('Multiple Modality were found')
    mod = mod_list[0]

    if mod == 'CR':
        if len(index_list) == 0:
            raise AssertionError('No instance was found')
        elif len(index_list) > 1:
            raise AssertionError('Modality "CR": Multiple instances were found.')

        dcmObj = index_data_dict.get(index_list[0])
        chirality = None
        spacing_slice = None
        series_like = getattr(dcmObj, 'ViewPosition', None)
        image_orientation_patient = None
    elif mod == 'DX':
        if len(index_list) == 0:
            raise AssertionError('No instance was found')
        elif len(index_list) > 1:
            raise AssertionError('Modality "DX": Multiple instances were found.')

        dcmObj = index_data_dict.get(index_list[0])
        chirality = None
        spacing_slice = None
        series_like = None # AP / PA / ???
        image_orientation_patient = None
    elif mod == 'CT':
        if len(index_list) < 2:
            raise AssertionError('Cannot be reconstructed: Number of CT images less than 2')
        res = _recon_ct(index_map, index_data_dict, index_list)
        index_data_dict = res['index_data_dict']
        index_list = res['index_list']
        index_map = res['index_map']
        chirality = res['chirality']
        spacing_slice = res['spacing_slice']
        series_like = res['series_like']
        image_orientation_patient = res['image_orientation_patient']
    else:
        raise AssertionError(f'No reconstruction methods for {mod}')

    # 6. retag contrast with ContrastBolusAgent: null -> w/o contrast, otherwise -> w/
    use_contrast = getattr(index_data_dict.get(index_list[0]), 'ContrastBolusAgent', False)
    use_contrast = (use_contrast not in [None, '', False])
    
    # 7. return map, dicom_dict that indexing by InstanceNumber, use_contrast, series_like, ..
    first = index_data_dict.get(index_list[0])
    index_dicom_dict = {i: dicom_dict.get(unikey) for i, unikey in index_map.items()}
    return {'index_list': index_list,
            'index_map': index_map,
            'index_dicom_dict': index_dicom_dict,
            'chirality': chirality,
            'series_like': series_like,
            'use_contrast': use_contrast,
            'spacing_slice': spacing_slice,
            'ImageOrientationPatient': image_orientation_patient,
            'N_pixel_i': getattr(first, 'Rows', -1),
            'N_pixel_j': getattr(first, 'Columns', -1),
            'spacing_i': float(getattr(first, 'PixelSpacing', [1, 1])[1]), # i -> spacing between columns
            'spacing_j': float(getattr(first, 'PixelSpacing', [1, 1])[0]), # j -> spacing between rows
            'description': getattr(first, 'SeriesDescription', 'DefaultForNone'),
            'slice_thickness': getattr(first, 'SliceThickness', None),
            'patient_position': getattr(first, 'PatientPosition', None),
            }


def _recon_ct(index_map, index_data_dict, index_list):
    # for CT
    # 2. check inner product of ImageOrientationPatient of first and second images: if == 0 -> discard first
    #    2.1 get first and second
    dcm1 = index_data_dict.get(index_list[0])
    dcm2 = index_data_dict.get(index_list[1])
    #    2.2 get ImageOrientationPatient and get inner product
    iop1 = getattr(dcm1, 'ImageOrientationPatient', None)
    if iop1 is None:
        raise AssertionError('ImageOrientationPatient not found')
    iop2 = getattr(dcm2, 'ImageOrientationPatient', None)
    if iop2 is None:
        raise AssertionError('ImageOrientationPatient not found')
    iop1 = [float(x) for x in iop1]
    iop2 = [float(x) for x in iop2]
    dot = (  (sum([iop1[i]*iop2[i] for i in range(3)]) * sum([iop1[i+3]*iop2[i+3] for i in range(3)]))
           - (sum([iop1[i]*iop2[i+3] for i in range(3)]) * sum([iop1[i+3]*iop2[i] for i in range(3)])))
    if abs(dot) < 0.1:
        # discard first
        index_map = {k:v for k, v in index_map.items() if k != index_list[0]}
        index_data_dict = {k:v for k, v in index_data_dict.items() if k != index_list[0]}
        index_list = list(index_map.keys())
        index_list.sort()

    # 3. check alignment from ImagePositionPatient 
    #    3.1 check image size
    if len(set([x.Columns for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('Columns of images mismatch')
    if len(set([x.Rows for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('Rows of images mismatch')
    #    3.2 check pixel spacing in precision of 0.001
    if len(set([round(float(x.PixelSpacing[0])/1E-3) for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('PixelSpacing of columns mismatch')
    if len(set([round(float(x.PixelSpacing[1])/1E-3) for _, x in index_data_dict.items()])) > 1:
        raise AssertionError('PixelSpacing of columns mismatch')
    #    3.2 check slice spacing in precision of 0.01
    okay = True
    spacing_list = list()
    iop_list = [[float(x)
                 for x in index_data_dict.get(ind).ImageOrientationPatient]
                for ind in index_list]
    pos_list = [[float(x)
                 for x in index_data_dict.get(ind).ImagePositionPatient]
                for ind in index_list]
    chirality_list = list()
    msglist = list()
    for i, ind1 in enumerate(index_list):
        if i == 0:
            continue

        ind0 = index_list[i-1]
        if (ind1 - ind0) != 1:
            okay = False
            msglist.append(f'index jump {ind0} -> {ind1}')
            continue
        iop = iop_list[i-1]
        pos1 = pos_list[i]
        pos0 = pos_list[i-1]
        dp = [pos1[j] - pos0[j] for j in range(3)]
        spacing = sum([x**2 for x in dp])**0.5
        if abs(spacing) < 1e-4:
            warnings.warn(f'spacing between {ind0} and {ind1} is 0')
            continue
        tmp_dot = sum([
            dp[0] *iop[1]*iop[3+2],
            dp[1] *iop[2]*iop[3+0],
            dp[2] *iop[0]*iop[3+1],
            -dp[0]*iop[2]*iop[3+1],
            -dp[1]*iop[0]*iop[3+2],
            -dp[2]*iop[1]*iop[3+0]
            ])
        if abs(tmp_dot/spacing) < 0.99:
            msglist.append(f'{ind0} and {ind1} do not aligned')
            okay = False
            continue
        chirality_list.append(tmp_dot/spacing>0)
        spacing_list.append(spacing)

    if not okay:
        raise AssertionError('\n'.join(msglist))
    if len(set(chirality_list)) > 1:
        raise AssertionError('Direction (Chirality) of images mismatch')
    elif len(chirality_list) == 0:
        chirality = None
    else:
        chirality = chirality_list[0]

    # 4. calculate spacing
    # if len(set([round(x*1E3) for x in spacing_list])) != 1:
        # raise AssertionError('Slice Spacing mismatch')
    # spacing = abs(spacing_list[0])
    # 5. retag series with ImageOrientationPatient: Axial-like, Coronal-like, Sagittal-like, Other
    iop = iop_list[0]
    k_like = {'axial': [0,0,1], 'coronal': [0,1,0], 'sagittal': [1,0,0]}
    series_like = 'other'
    for cls, vec in k_like.items():
        if (abs(sum([iop[i]  *vec[i] for i in range(3)])) < 0.1 and
            abs(sum([iop[i+3]*vec[i] for i in range(3)])) < 0.1):
            series_like = cls
            break

    return {'index_data_dict': index_data_dict,
            'index_list': index_list,
            'index_map': index_map,
            'chirality': chirality,
            # 'spacing_slice': float('%.3f' % spacing),
            'spacing_slice': set([round(x*1E3) for x in spacing_list]),
            'series_like': series_like,
            'image_orientation_patient': [float('%.4f'%x) for x in iop],
            }


def get_instance_info(dcmObj):
    STORE_VAR = ['Modality', 'PatientID', 'AccessionNumber',
                 'StudyDate', 'StudyTime', 'StudyDescription',
                 'SeriesDate', 'SeriesTime', 'SeriesDescription',
                 'SamplesPerPixel', 'Rows', 'Columns', 'PixelSpacing',
                 'SliceThickness', 'SpacingBetweenSlices', 'SliceLocation',
                 'PhotometricInterpretation',
                 'BitsAllocated', 'BitsStored', 'HighBit', 'PixelRepresentation',
                 'PlanarConfiguration', 'PixelAspectRatio', 'SmallestImagePixelValue',
                 'LargestImagePixelValue', 'ColorSpace', 'BodyPartExamined',
                 'PatientPosition', 'ImageOrientationPatient', 'ImagePositionPatient',
                 'PatientOrientation', 'AnatomicalOrientationType', 'ViewPosition'
                 ]
    res = dict()
    res['bits_origin'] = getattr(dcmObj, 'BitsStored', None)
    image_position = getattr(dcmObj, 'ImagePositionPatient', None)
    if image_position is not None:
        image_position = [float('%.4f'%float(x)) for x in image_position]
    res['image_position'] = image_position
    return res
