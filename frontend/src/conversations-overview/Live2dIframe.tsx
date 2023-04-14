import { apiURL } from '../Settings/Constants';

interface ILive2dIframeProps {
    session_token: string;
}

const Live2dIframe = ({ session_token }: ILive2dIframeProps) => {
    return (
        <iframe
            className="live2diframe"
            title="Live 2D canvas"
            src={apiURL + "/live2d/Samples/TypeScript/Demo/index.html?sessiontoken=" + session_token}
            style={{ width: '100%', height: '99%', border: 'none' }}
        />
    );
};

export default Live2dIframe;